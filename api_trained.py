"""
HTTP API: загрузка PDF / Word (docx) / Excel / txt / csv → JSON с типом документа и полями.

- Классификатор — дообученный чекпоинт `doc_classifier`.
- Поля: **`API_FIELDS_MODE=neural_trained`** — извлечение полей **только seq2seq**, без regex/эвристик
  (если модель ошиблась или пусто — пустые поля). Режим **merge** — seq2seq + regex как подстраховка (по умолчанию).
  **regex_only** — только эвристики, без нейросети полей (если нет чекпоинта извлечения).

Переменная окружения `API_FIELDS_MODE`: `merge` (по умолчанию) | `neural_trained` (только нейросеть полей) | `regex_only`.

Локальные секреты: файл **`.env`** в корне проекта (рядом с `api_trained.py`) подхватывается при старте.

Запуск: py -3 api_trained.py
Или: py -3 -m uvicorn api_trained:app --host 127.0.0.1 --port 8080

Нагрузка: десятки запросов в сутки — обычно достаточно одного процесса uvicorn (модели в памяти).
Несколько воркеров uvicorn не поднимайте без необходимости: каждый дублирует веса в RAM.
Инференс выполняется в пуле потоков, чтобы не блокировать остальные HTTP-запросы.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

# До любого os.environ.get — чтобы подтянуть OCR_YANDEX_*, API_* и т.д.
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

import logging
import os
import tempfile
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoConfig

from document_classifier.config import DEFAULT_MODEL_NAME
from document_classifier.inference import DocumentClassifier
from document_classifier.extract import ocr_yandex_bytes_for_test
from document_classifier.neural_extract import extract_backend_is_vllm

# Базовая seq2seq из train_extract.py (если в config нет явной ссылки)
DEFAULT_EXTRACT_BASE_MODEL = "google/flan-t5-small"

DEFAULT_CLASSIFIER = Path(os.environ.get("DOC_CLASSIFIER_CKPT", "checkpoints/doc_classifier"))
DEFAULT_EXTRACT = Path(os.environ.get("INVOICE_EXTRACT_CKPT", "checkpoints/invoice_extract"))
MAX_UPLOAD_BYTES = int(os.environ.get("API_MAX_UPLOAD_MB", "32")) * 1024 * 1024

# merge | neural_trained | regex_only (только эвристики полей, без seq2seq)
_FIELDS_MODE_RAW = os.environ.get("API_FIELDS_MODE", "merge").strip().lower()
if _FIELDS_MODE_RAW not in ("merge", "neural_trained", "regex_only"):
    raise RuntimeError(
        "API_FIELDS_MODE должен быть 'merge', 'neural_trained' или 'regex_only', "
        f"получено: {_FIELDS_MODE_RAW!r}",
    )
API_FIELDS_MODE = _FIELDS_MODE_RAW

logger = logging.getLogger("api_trained")


class TextExtractMode(str, Enum):
    """Как получить сырой текст до классификации/полей (для PDF и изображений)."""

    auto = "auto"
    local = "local"
    ocr = "ocr"


def _setup_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


app = FastAPI(
    title="Классификация и извлечение реквизитов",
    version="1.0.0",
    description=(
        "Поля: neural_trained — только seq2seq; merge — seq2seq + regex; "
        "regex_only — только эвристики. "
        f"Текущий режим: {API_FIELDS_MODE}."
    ),
)

_clf: Optional[DocumentClassifier] = None
_PIPELINE_META: Optional[NeuralPipelineInfo] = None


def get_classifier() -> DocumentClassifier:
    global _clf
    if _clf is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return _clf


def _safe_architectures(cfg: Any) -> Optional[str]:
    arch = getattr(cfg, "architectures", None)
    if isinstance(arch, list) and arch:
        return str(arch[0])
    return None


class NeuralModelInfo(BaseModel):
    task: str
    checkpoint: str
    pretrained_basis: str
    model_type: str = ""
    architecture: Optional[str] = None
    num_labels: Optional[int] = None


class NeuralPipelineInfo(BaseModel):
    summary: str
    fields_mode: str
    field_extraction_note: str
    document_classifier: NeuralModelInfo
    field_extraction: NeuralModelInfo


def _build_pipeline_meta() -> NeuralPipelineInfo:
    clf_path = str(DEFAULT_CLASSIFIER.resolve())
    ext_path = str(DEFAULT_EXTRACT.resolve())
    try:
        cc = AutoConfig.from_pretrained(clf_path)
        clf_type = getattr(cc, "model_type", None) or "unknown"
        clf_arch = _safe_architectures(cc)
        n_lab = getattr(cc, "num_labels", None)
    except Exception:
        clf_type, clf_arch, n_lab = "unknown", None, None

    if API_FIELDS_MODE == "regex_only":
        ext_type, ext_arch = "n/a", None
        merge_note = (
            "Только эвристики (extract_invoice_fields): regex и правила. "
            "Нейросеть извлечения полей не загружается и не вызывается."
        )
        ext_checkpoint = "(не используется)"
        ext_basis = "Режим API_FIELDS_MODE=regex_only — seq2seq не применяется."
        ext_task = "эвристики: регулярные выражения (invoice_fields)"
    else:
        if extract_backend_is_vllm():
            vbase = (os.environ.get("VLLM_OPENAI_BASE") or "").strip()
            vmodel = (os.environ.get("VLLM_MODEL") or "").strip()
            ext_type, ext_arch = "vllm", None
            merge_note = (
                "Нейросеть (seq2seq через vLLM) для полей + эвристики regex, если модель не заполнила поле."
                if API_FIELDS_MODE == "merge"
                else "Только seq2seq через vLLM для полей, без regex."
            )
            ext_checkpoint = f"{vbase} (model={vmodel})" if vbase else "(vLLM)"
            ext_basis = (
                "Инференс через OpenAI-совместимый API vLLM на удалённом сервере; "
                "локальный каталог seq2seq не используется."
            )
            ext_task = "seq2seq (vLLM): текст счёта → JSON реквизитов"
        else:
            try:
                ec = AutoConfig.from_pretrained(ext_path)
                ext_type = getattr(ec, "model_type", None) or "unknown"
                ext_arch = _safe_architectures(ec)
            except Exception:
                ext_type, ext_arch = "unknown", None

            merge_note = (
                "Нейросеть (seq2seq) для полей + эвристики regex, если модель не заполнила поле."
                if API_FIELDS_MODE == "merge"
                else "Только seq2seq для полей, без regex."
            )
            ext_checkpoint = ext_path
            ext_basis = (
                f"База: предобученная seq2seq (типичный старт: {DEFAULT_EXTRACT_BASE_MODEL}), "
                "дообучение на data/extract_train.jsonl."
            )
            ext_task = "seq2seq: текст счёта → JSON реквизитов"

    summary = (
        "Тип документа — дообученный классификатор. "
        + (
            "Реквизиты — только эвристики по тексту (без seq2seq)."
            if API_FIELDS_MODE == "regex_only"
            else (
                "Реквизиты — seq2seq через vLLM и/или эвристики (см. field_extraction_note)."
                if extract_backend_is_vllm()
                else "Реквизиты — дообученная seq2seq и/или эвристики (см. field_extraction_note)."
            )
        )
    )

    return NeuralPipelineInfo(
        summary=summary,
        fields_mode=API_FIELDS_MODE,
        field_extraction_note=merge_note,
        document_classifier=NeuralModelInfo(
            task="классификация типа первичного документа",
            checkpoint=clf_path,
            pretrained_basis=(
                f"База обучения: предобученная модель семейства BERT/RuBERT "
                f"(типичный старт: {DEFAULT_MODEL_NAME}), затем дообучение на размеченных jsonl/папках."
            ),
            model_type=str(clf_type),
            architecture=clf_arch,
            num_labels=int(n_lab) if isinstance(n_lab, int) else None,
        ),
        field_extraction=NeuralModelInfo(
            task=ext_task,
            checkpoint=ext_checkpoint,
            pretrained_basis=ext_basis,
            model_type=str(ext_type),
            architecture=ext_arch,
            num_labels=None,
        ),
    )


@app.on_event("startup")
def _load_models() -> None:
    global _clf, _PIPELINE_META
    _setup_logging()
    # Ускорение извлечения по умолчанию (можно переопределить переменными окружения)
    os.environ.setdefault("EXTRACT_NUM_BEAMS", "1")
    os.environ.setdefault("EXTRACT_MAX_NEW_TOKENS", "320")
    if not DEFAULT_CLASSIFIER.is_dir():
        raise RuntimeError(
            f"Нет чекпоинта классификатора: {DEFAULT_CLASSIFIER}. Обучите: py -3 train.py ...",
        )
    use_neural_fields = API_FIELDS_MODE != "regex_only"
    if use_neural_fields and not extract_backend_is_vllm() and not (DEFAULT_EXTRACT / "config.json").is_file():
        raise RuntimeError(
            f"Нет чекпоинта извлечения: {DEFAULT_EXTRACT} (нужен config.json). "
            "Обучите: py -3 train_extract.py — или подключите vLLM (VLLM_OPENAI_BASE + VLLM_MODEL, либо EXTRACT_BACKEND=vllm), "
            "или задайте API_FIELDS_MODE=regex_only (только эвристики).",
        )
    _clf = DocumentClassifier(
        DEFAULT_CLASSIFIER,
        extract_checkpoint=DEFAULT_EXTRACT if use_neural_fields else None,
        use_neural_extract=use_neural_fields,
        fields_mode=API_FIELDS_MODE,
    )
    _PIPELINE_META = _build_pipeline_meta()


class PredictResponse(BaseModel):
    label: str = ""
    fields: Dict[str, Any] = Field(default_factory=dict)
    extracted_chars: int = 0
    error: Optional[str] = None
    source_text_preview: Optional[str] = None
    # Полный извлечённый текст (для UI подсветки); при include_source_text=0 не отдаётся
    source_text: Optional[str] = None
    neural_pipeline: NeuralPipelineInfo
    # мс: extract_text, classify, fields, total (инференс), upload, api_total (весь запрос)
    timing_ms: Optional[Dict[str, Any]] = None


class OcrTestResponse(BaseModel):
    ok: bool
    ocr_text: str = ""
    chars: int = 0
    timing_ms: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


def _result_to_response(raw: Dict[str, Any], *, include_source_text: bool = False) -> PredictResponse:
    if _PIPELINE_META is None:
        raise HTTPException(status_code=503, detail="Метаданные пайплайна не инициализированы")
    st = raw.get("source_text") if include_source_text else None
    if isinstance(st, str) and not st.strip():
        st = None
    return PredictResponse(
        label=str(raw.get("label") or ""),
        fields=raw.get("fields") or {},
        extracted_chars=int(raw.get("extracted_chars") or 0),
        error=(raw.get("error") or None) or None,
        source_text_preview=raw.get("source_text_preview"),
        source_text=st if include_source_text else None,
        neural_pipeline=_PIPELINE_META,
        timing_ms=raw.get("timing_ms"),
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "ok", "fields_mode": API_FIELDS_MODE}
    if _PIPELINE_META is not None:
        out["neural_pipeline"] = _PIPELINE_META.model_dump()
    return out


@app.post("/ocr/test", response_model=OcrTestResponse)
async def ocr_test(file: UploadFile = File(...)) -> OcrTestResponse:
    """
    Проверка токена/папки Yandex OCR.
    Принимает .pdf/.png/.jpg/.jpeg и возвращает распознанный текст (только OCR).
    """
    name = file.filename or "upload"
    t0 = time.perf_counter()
    body = await file.read()
    t1 = time.perf_counter()
    try:
        txt = await asyncio.to_thread(ocr_yandex_bytes_for_test, name, body)
        t2 = time.perf_counter()
        return OcrTestResponse(
            ok=True,
            ocr_text=txt,
            chars=len(txt or ""),
            timing_ms={
                "upload": round((t1 - t0) * 1000, 2),
                "ocr_total": round((t2 - t1) * 1000, 2),
                "api_total": round((t2 - t0) * 1000, 2),
            },
        )
    except Exception as e:
        t2 = time.perf_counter()
        return OcrTestResponse(
            ok=False,
            ocr_text="",
            chars=0,
            timing_ms={
                "upload": round((t1 - t0) * 1000, 2),
                "api_total": round((t2 - t0) * 1000, 2),
            },
            error=str(e),
        )


async def _predict_one_upload(
    file: UploadFile,
    *,
    include_source_text: bool = False,
    text_extract_mode: TextExtractMode = TextExtractMode.auto,
) -> PredictResponse:
    """Один файл из multipart — общая логика для POST /predict."""
    name = file.filename or "upload"
    suf = Path(name).suffix.lower()
    allowed = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv", ".png", ".jpg", ".jpeg"}
    if suf not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Файл {name!r}: неподдерживаемое расширение {suf!r}. Допустимо: {', '.join(sorted(allowed))}",
        )

    t0 = time.perf_counter()
    body = await file.read()
    t_after_read = time.perf_counter()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Файл {name!r} больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ",
        )

    clf = get_classifier()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)

    try:
        raw = await asyncio.to_thread(
            clf.predict_file,
            tmp_path,
            5,
            text_extract_mode=text_extract_mode.value,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    t_end = time.perf_counter()
    timing = dict(raw.get("timing_ms") or {})
    timing["upload"] = round((t_after_read - t0) * 1000, 2)
    timing["api_total"] = round((t_end - t0) * 1000, 2)
    raw["timing_ms"] = timing

    logger.info(
        "predict file=%s text_mode=%s upload_ms=%.2f extract_ms=%.2f classify_ms=%.2f fields_ms=%.2f "
        "inference_total_ms=%.2f api_total_ms=%.2f",
        name,
        text_extract_mode.value,
        timing.get("upload", 0.0),
        timing.get("extract_text", 0.0),
        timing.get("classify", 0.0),
        timing.get("fields", 0.0),
        timing.get("total", 0.0),
        timing.get("api_total", 0.0),
    )

    return _result_to_response(raw, include_source_text=include_source_text)


@app.get("/recognition")
def recognition_ui() -> FileResponse:
    """Веб-интерфейс: параметры + подсветка значений в тексте при наведении."""
    p = Path(__file__).resolve().parent / "static" / "recognition.html"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="static/recognition.html не найден")
    return FileResponse(p, media_type="text/html; charset=utf-8")


@app.post("/predict")
async def predict(
    file: List[UploadFile] = File(...),
    include_source_text: bool = Query(
        False,
        description="Включить поле source_text (полный текст документа) для UI подсветки",
    ),
    text_extract_mode: TextExtractMode = Query(
        TextExtractMode.auto,
        description=(
            "Извлечение текста до моделей: auto — при настроенном Yandex Vision PDF как JPEG (OCR страниц); "
            "если OCR недоступен — текстовый слой PDF; docx/xlsx/xls/txt/csv всегда локально; "
            "local — только PyMuPDF/pypdf и Office/текст, без Yandex (для картинок недоступно); "
            "ocr — принудительно Yandex Vision для PDF и изображений (ошибка при сбое OCR)"
        ),
    ),
) -> Union[PredictResponse, List[PredictResponse]]:
    """
    Form-data: одно или несколько полей **file** — `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`, `.csv`,
    а также `.png`/`.jpg`/`.jpeg` (OCR; в режиме `local` для картинок вернётся ошибка).
    PDF и изображения: режим `auto`/`ocr` управляет Yandex OCR; Word/Excel/txt/csv читаются локально.

    Query **text_extract_mode**: см. описание параметра (auto / local / ocr).

    - **Один файл** → ответ — один JSON-объект (`PredictResponse`).
    - **Несколько файлов** → ответ — JSON-массив объектов в том же порядке.
    """
    if not file:
        raise HTTPException(status_code=400, detail="Передайте хотя бы один файл в полях file")
    if len(file) == 1:
        return await _predict_one_upload(
            file[0],
            include_source_text=include_source_text,
            text_extract_mode=text_extract_mode,
        )
    return [
        await _predict_one_upload(
            f,
            include_source_text=include_source_text,
            text_extract_mode=text_extract_mode,
        )
        for f in file
    ]


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8080"))
    uvicorn.run("api_trained:app", host=host, port=port, reload=False)
