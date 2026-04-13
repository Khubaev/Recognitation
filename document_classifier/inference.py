from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DOC_LABELS
from .extract import MAX_DOC_CHARS, extract_text_from_file
from .extract_target import merge_extracted
from .invoice_fields import (
    FIELD_LABELS,
    enrich_fields_from_regex_fallback,
    extract_invoice_fields,
    project_public_fields,
)
from .neural_extract import NeuralFieldExtractor

MAX_INFER_CHARS = 8000


DEFAULT_EXTRACT_CKPT = Path("checkpoints/invoice_extract")

_log = logging.getLogger(__name__)


def _empty_fields() -> Dict[str, Any]:
    """Пустая структура полей (без эвристик)."""
    return {k: "" for k in FIELD_LABELS} | {"items": []}


def _neural_output_empty(fields: Dict[str, Any]) -> bool:
    """Нет ни одного заполненного поля и ни одной строки в items."""
    for k in FIELD_LABELS:
        v = fields.get(k)
        if isinstance(v, str) and v.strip():
            return False
        if v is not None and not isinstance(v, str) and str(v).strip():
            return False
    items = fields.get("items")
    if isinstance(items, list) and len(items) > 0:
        return False
    return True


class DocumentClassifier:
    def __init__(
        self,
        checkpoint_dir: Path,
        device: Optional[str] = None,
        *,
        extract_checkpoint: Optional[Path] = None,
        use_neural_extract: bool = True,
        fields_mode: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {self.checkpoint_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_dir))
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        cfg = self.model.config
        self.id2label = {int(k): v for k, v in cfg.id2label.items()} if cfg.id2label else {
            i: DOC_LABELS[i] for i in range(len(DOC_LABELS))
        }

        self._neural: Optional[NeuralFieldExtractor] = None
        if use_neural_extract:
            ec = extract_checkpoint
            if ec is None and DEFAULT_EXTRACT_CKPT.is_dir():
                ec = DEFAULT_EXTRACT_CKPT
            if ec is not None and Path(ec).is_dir():
                try:
                    if (Path(ec) / "config.json").exists():
                        self._neural = NeuralFieldExtractor(Path(ec), device=str(self.device))
                except Exception:
                    self._neural = None

        # merge | neural_only | neural_trained | regex_only
        # neural_trained — только дообученный seq2seq, без regex и без merge
        if fields_mode is not None and fields_mode not in (
            "merge",
            "neural_only",
            "neural_trained",
            "regex_only",
        ):
            raise ValueError("fields_mode: merge | neural_only | neural_trained | regex_only")
        if fields_mode is None:
            self.fields_mode = "merge"
        else:
            self.fields_mode = fields_mode

        if self.fields_mode == "neural_trained" and not self._neural:
            raise ValueError(
                "Режим neural_trained требует чекпоинт извлечения (seq2seq): "
                "каталог с config.json, например checkpoints/invoice_extract.",
            )

    def _fields_for_text(self, text: str) -> Dict[str, Any]:
        if self.fields_mode == "neural_trained":
            assert self._neural is not None
            try:
                return self._neural.extract(text)
            except Exception:
                return _empty_fields()

        regex_f = extract_invoice_fields(text)
        if self.fields_mode == "regex_only":
            return regex_f
        if self.fields_mode == "neural_only":
            if not self._neural:
                return regex_f
            try:
                neural_out = self._neural.extract(text)
                # Нейросеть могла вернуть пустой/битый JSON — подставляем эвристики
                if _neural_output_empty(neural_out):
                    return merge_extracted(neural_out, regex_f)
                return neural_out
            except Exception:
                return regex_f
        if self._neural:
            try:
                return merge_extracted(self._neural.extract(text), regex_f)
            except Exception:
                return regex_f
        return regex_f

    def predict_text(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        full = text or ""
        text = full[:MAX_INFER_CHARS]
        t0 = time.perf_counter()
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        order = probs.argsort(descending=True).tolist()[:top_k]
        top: List[Tuple[str, float]] = []
        for i in order:
            name = self.id2label.get(i, DOC_LABELS[i] if i < len(DOC_LABELS) else str(i))
            top.append((name, float(probs[i].item())))
        best_i = int(probs.argmax())
        best_name = self.id2label.get(best_i, DOC_LABELS[best_i] if best_i < len(DOC_LABELS) else str(best_i))
        t1 = time.perf_counter()
        fields = self._fields_for_text(full)
        fields = enrich_fields_from_regex_fallback(full, fields)
        fields = project_public_fields(fields)
        t2 = time.perf_counter()
        classify_ms = round((t1 - t0) * 1000, 2)
        fields_ms = round((t2 - t1) * 1000, 2)
        return {
            "label": best_name,
            "confidence": float(probs[best_i].item()),
            "top": top,
            "fields": fields,
            "timing_ms": {
                "classify": classify_ms,
                "fields": fields_ms,
            },
        }

    def predict_file(
        self,
        path: Path,
        top_k: int = 5,
        *,
        text_extract_mode: str = "auto",
    ) -> Dict[str, Any]:
        path = Path(path)
        t_all = time.perf_counter()
        try:
            t0 = time.perf_counter()
            text = extract_text_from_file(path, text_extract_mode=text_extract_mode)
            t1 = time.perf_counter()
            extract_ms = round((t1 - t0) * 1000, 2)
        except Exception as e:
            return {
                "label": "",
                "confidence": 0.0,
                "top": [],
                "error": str(e),
                "source_text_preview": "",
                "source_text": "",
                "extracted_chars": 0,
                "fields": project_public_fields({k: "" for k in FIELD_LABELS} | {"items": []}),
                "timing_ms": {"extract_text": 0.0, "total": round((time.perf_counter() - t_all) * 1000, 2)},
            }
        out = self.predict_text(text, top_k=top_k)
        inner = out.pop("timing_ms", {})
        out["source_text_preview"] = text[:2000] if text else ""
        out["source_text"] = (text or "")[:MAX_DOC_CHARS]
        out["extracted_chars"] = len(text)
        out["error"] = ""
        if not text.strip():
            out["error"] = "Пустой текст (скан без OCR?)"
        total_ms = round((time.perf_counter() - t_all) * 1000, 2)
        out["timing_ms"] = {
            "extract_text": extract_ms,
            **inner,
            "total": total_ms,
        }
        _log.info(
            "recognition path=%s chars=%s extract_ms=%.2f classify_ms=%.2f fields_ms=%.2f total_ms=%.2f",
            path.name,
            len(text),
            extract_ms,
            inner.get("classify", 0.0),
            inner.get("fields", 0.0),
            total_ms,
        )
        return out
