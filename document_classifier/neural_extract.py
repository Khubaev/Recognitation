"""Дообученная seq2seq-модель: текст счёта → JSON полей.

Ускорение (переменные окружения, без правки кода):
- EXTRACT_NUM_BEAMS — по умолчанию 1 (жадный декодинг; было 2 — ~до 2× медленнее).
- EXTRACT_MAX_NEW_TOKENS — макс. длина ответа (по умолчанию 320).
- EXTRACT_MAX_INPUT_CHARS — обрезка текста документа (по умолчанию 6000).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .extract_target import normalize_parsed, parse_model_json

EXTRACT_PREFIX = (
    "Извлеки в JSON реквизиты счёта на оплату. В первую очередь заполни: "
    "ИНН поставщика, Поставщик (наименование организации), Номер счета, Дата счета, Итого (сумма). "
    "Дополнительно при наличии в тексте: "
    "Банк получателя, Получатель, КПП поставщика, ИНН покупателя, КПП покупателя, "
    "БИК, Счет, Покупатель, Товары, Количество, Ед.Из, Цена, Сумма; "
    "items — массив объектов {name,qty,unit,price,sum}. Текст документа:\n"
)


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
