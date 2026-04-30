"""Дообученная seq2seq-модель: текст счёта → JSON полей.

Локально: transformers + чекпоинт в каталоге.

Удалённый vLLM (OpenAI-совместимый API):
- Если заданы **оба** ``VLLM_OPENAI_BASE`` и ``VLLM_MODEL`` — извлечение сразу идёт в vLLM (без отдельного флага).
- Либо явно ``EXTRACT_BACKEND=vllm`` (удобно, когда URL/модель задаёте иначе).
- Чтобы **принудительно** локальный чекпоинт при наличии ``VLLM_*`` в окружении: ``EXTRACT_BACKEND=local`` или ``transformers``.

Переменные окружения (локальный инференс):
- EXTRACT_NUM_BEAMS — по умолчанию 1 (жадный декодинг; было 2 — ~до 2× медленнее).
- EXTRACT_MAX_NEW_TOKENS — макс. длина ответа (по умолчанию 320).
- EXTRACT_MAX_INPUT_CHARS — обрезка текста документа (по умолчанию 6000).

vLLM: ``VLLM_TIMEOUT_SEC`` (сек., по умолчанию 120), ``VLLM_API_KEY`` (Bearer, если нужен),
``VLLM_MAX_INPUT_CHARS`` (по умолчанию 30000; отдельный лимит для LLM-ветки),
``VLLM_MAX_TOKENS`` (макс. длина ответа для vLLM, по умолчанию берётся из EXTRACT_MAX_NEW_TOKENS).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .extract_target import normalize_parsed, parse_model_json

_log = logging.getLogger(__name__)

# JSON Schema для guided_json (constrained decoding в vLLM)
EXTRACT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "Дата счета":             {"type": "string"},
        "Номер счета":            {"type": "string"},
        "ИНН поставщика":         {"type": "string"},
        "Поставщик":                {"type": "string"},
        "Итого":                  {"type": "string"},
    },
    "required": [
        "Дата счета", "Номер счета", "ИНН поставщика",
        "Поставщик", "Итого",
    ],
    "additionalProperties": False,
}

EXTRACT_PREFIX = (
    "Извлеки из документа (счёт, накладная, УПД) один JSON-объект. Нужны только поля: "
    "\"Дата счета\", \"Номер счета\", \"ИНН поставщика\", "
    "\"Поставщик\" — только название организации (ООО/АО/ИП и т.п.) из шапки документа, "
    "в блоке где указан ИНН поставщика или у реквизитов «Продавец»/«Поставщик»/«Получатель платежа». "
    "Если явного «Поставщик» нет — возьми организацию из блока «Продавец»/«Исполнитель»/«Получатель», "
    "где рядом указан ИНН поставщика. "
    "\"Итого\" — полная сумма счёта цифрами из строки «Итого к оплате» / «Всего на сумму» (например 7850,00 или 7 850,00). "
    "Если таких строк нет — ищи сумму в назначении платежа или в строке «В т.ч. НДС» (берётся сумма включая НДС, а не сам НДС). "
    "Не подставляй сумму прописью («семь тысяч … рублей»). "
    "Не используй юридические фразы из условий оферты: «обязуется поставить», «обязуется оплатить». "
    "Не используй адресные фрагменты вместо названия организации: «ул./улица», «дом», «корпус», «помещение». "
    "В «Поставщик» возвращай именно юрлицо/ИП (ООО/АО/ИП/ПАО и т.п.) рядом с ИНН поставщика. "
    "Не подставляй сюда строки из таблицы товаров, артикулы, подписи колонок. "
    "«Номер счета» — из заголовка «Счёт на оплату № …»; не путай с банковским «Сч. №» (длинное число 18–22 цифры). "
    "Другие ключи не добавляй. Значения — строки; если поля нет в тексте — \"\".\n"
    "\n"
    "Пример 1:\n"
    "Текст: Счёт на оплату № 47 от 12.02.2024\n"
    "Поставщик: ООО \"Альфа Трейд\" ИНН 7701234567 КПП 770101001\n"
    "Итого к оплате: 45 600,00 руб.\n"
    "Ответ: {\"Дата счета\": \"12.02.2024\", \"Номер счета\": \"47\", "
    "\"ИНН поставщика\": \"7701234567\", \"Поставщик\": \"ООО \\\"Альфа Трейд\\\"\", "
    "\"Итого\": \"45 600,00\"}\n"
    "\n"
    "Пример 2:\n"
    "Текст: СЧЁТ № 2024-003 от 05.03.2024\n"
    "Исполнитель: ИП Иванов Сергей Николаевич, ИНН 501234567890\n"
    "Всего на сумму: 12 000 руб. 00 коп.\n"
    "Ответ: {\"Дата счета\": \"05.03.2024\", \"Номер счета\": \"2024-003\", "
    "\"ИНН поставщика\": \"501234567890\", \"Поставщик\": \"ИП Иванов Сергей Николаевич\", "
    "\"Итого\": \"12 000,00\"}\n"
    "\n"
    "Пример 3:\n"
    "Текст: КОМПЛЕКСНЫЕ ПОСТАВКИ ЭЛЕКТРОТЕХНИЧЕСКОГО ОБОРУДОВАНИЯ\n"
    "ИНН 7842224734 КПП 784201001\n"
    "Получатель\n"
    "АО \"ТД \"Электротехмонтаж\"\n"
    "Сч. № 40702810190330002617\n"
    "Назначение платежа: Оплата за товар по сч. N 519/4102959 от 13.04.2026\n"
    "В т.ч. НДС – 52587,49 руб\n"
    "Покупатель: ООО СМК \"ВЫСОТА\" ИНН 7453330144\n"
    "Ответ: {\"Дата счета\": \"13.04.2026\", \"Номер счета\": \"519/4102959\", "
    "\"ИНН поставщика\": \"7842224734\", \"Поставщик\": \"АО \\\"ТД \\\"Электротехмонтаж\\\"\", "
    "\"Итого\": \"52587,49\"}\n"
    "\n"
    "Текст документа:\n"
)


def extract_backend_is_vllm() -> bool:
    """
    vLLM: явно ``EXTRACT_BACKEND=vllm``, либо непустые пара ``VLLM_OPENAI_BASE`` + ``VLLM_MODEL``.
    ``EXTRACT_BACKEND=local`` / ``transformers`` — всегда локальный seq2seq из чекпоинта.
    """
    backend = os.environ.get("EXTRACT_BACKEND", "").strip().lower()
    if backend in ("local", "transformers"):
        return False
    if backend == "vllm":
        return True
    base = (os.environ.get("VLLM_OPENAI_BASE") or "").strip()
    model = (os.environ.get("VLLM_MODEL") or "").strip()
    return bool(base and model)


def _openai_chat_content(data: Any) -> str:
    """Текст ответа из JSON OpenAI chat/completions."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0]
    if not isinstance(c0, dict):
        return ""
    msg = c0.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            return c
    t = c0.get("text")
    if isinstance(t, str):
        return t
    return ""


class VllmNeuralFieldExtractor:
    """Извлечение полей через vLLM (POST /v1/chat/completions)."""

    def __init__(self) -> None:
        self.base = (os.environ.get("VLLM_OPENAI_BASE") or "").strip().rstrip("/")
        self.model = (os.environ.get("VLLM_MODEL") or "").strip()
        if not self.base or not self.model:
            raise ValueError(
                "vLLM: задайте переменные окружения VLLM_OPENAI_BASE (например http://127.0.0.1:8000/v1) "
                "и VLLM_MODEL (имя модели, как в vLLM).",
            )
        self.api_key = (os.environ.get("VLLM_API_KEY") or "").strip()
        self.timeout = float(os.environ.get("VLLM_TIMEOUT_SEC", "120"))
        # Для vLLM используем отдельный лимит входа, чтобы не наследовать
        # seq2seq-ориентированное ограничение EXTRACT_MAX_INPUT_CHARS.
        self.max_input_chars = max(2000, int(os.environ.get("VLLM_MAX_INPUT_CHARS", "30000")))
        self._max_new_tokens = max(
            64,
            int(os.environ.get("VLLM_MAX_TOKENS", os.environ.get("EXTRACT_MAX_NEW_TOKENS", "320"))),
        )

    def extract(self, text: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        import requests

        mnt = self._max_new_tokens if max_new_tokens is None else max(64, int(max_new_tokens))
        text = (text or "")[: self.max_input_chars]
        prompt = EXTRACT_PREFIX + text
        url = f"{self.base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mnt,
            "temperature": 0,
            "guided_json": EXTRACT_JSON_SCHEMA,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        raw = _openai_chat_content(data)
        if not raw.strip():
            _log.warning("vLLM: пустой ответ choices: %s", json.dumps(data)[:500])
        parsed = parse_model_json(raw)
        return normalize_parsed(parsed)


class NeuralFieldExtractor:
    def __init__(self, checkpoint_dir: Path, device: Optional[str] = None):
        self.path = Path(checkpoint_dir)
        if not self.path.exists():
            raise FileNotFoundError(f"Нет чекпоинта извлечения: {self.path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.path))
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.max_input_chars = max(1000, int(os.environ.get("EXTRACT_MAX_INPUT_CHARS", "6000")))
        self._num_beams = max(1, int(os.environ.get("EXTRACT_NUM_BEAMS", "1")))
        self._max_new_tokens = max(64, int(os.environ.get("EXTRACT_MAX_NEW_TOKENS", "320")))

    def extract(self, text: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        mnt = self._max_new_tokens if max_new_tokens is None else max(64, int(max_new_tokens))
        text = (text or "")[: self.max_input_chars]
        prompt = EXTRACT_PREFIX + text
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen_kw: Dict[str, Any] = {
            "max_new_tokens": mnt,
            "num_beams": self._num_beams,
            "do_sample": False,
        }
        if self._num_beams > 1:
            gen_kw["early_stopping"] = True
        with torch.no_grad():
            ids = self.model.generate(**enc, **gen_kw)
        raw = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        parsed = parse_model_json(raw)
        return normalize_parsed(parsed)


def create_neural_field_extractor(
    extract_checkpoint: Optional[Path],
    *,
    device: str,
) -> Optional[Union[NeuralFieldExtractor, VllmNeuralFieldExtractor]]:
    """
    Локальный seq2seq или vLLM по переменным окружения.
    Если задан vLLM, локальный чекпоинт извлечения не загружается.
    """
    if extract_backend_is_vllm():
        ext = VllmNeuralFieldExtractor()
        _log.info(
            "Извлечение полей: vLLM model=%s base=%s",
            (os.environ.get("VLLM_MODEL") or "").strip(),
            (os.environ.get("VLLM_OPENAI_BASE") or "").strip(),
        )
        return ext

    ec = extract_checkpoint
    if ec is None:
        default = Path("checkpoints/invoice_extract")
        if default.is_dir():
            ec = default
    if ec is not None and Path(ec).is_dir() and (Path(ec) / "config.json").exists():
        try:
            return NeuralFieldExtractor(Path(ec), device=device)
        except Exception as e:
            _log.warning("Не удалось загрузить локальный чекпоинт извлечения %s: %s", ec, e)
            return None
    return None
