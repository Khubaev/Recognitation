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

vLLM: ``VLLM_TIMEOUT_SEC`` (сек., по умолчанию 120), ``VLLM_API_KEY`` (Bearer, если нужен).
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

EXTRACT_PREFIX = (
    "Извлеки из документа (счёт, накладная, УПД) один JSON-объект. Нужны только поля: "
    "\"ИНН поставщика\", \"Поставщик\", "
    "\"Дата счета\", \"Номер счета\", \"ИНН покупателя\", "
    "\"Покупатель\" (наименование покупателя), "
    "\"Итого\" — только сумма цифрами, как в строке «Итого к оплате» / «Всего на сумму» (например 7850,00 или 7 850,00); "
    "не подставляй сумму прописью («семь тысяч … рублей»). "
    "Правило для «Поставщик»: только наименование организации (ООО/АО/ИП и т.п.) из шапки документа, "
    "в блоке где указан ИНН поставщика или у реквизитов «Продавец»/«Поставщик»/«Получатель платежа». "
    "Не подставляй сюда строки из таблицы товаров, артикулы, «заказной товар», подписи колонок (Z,N и т.п.). "
    "Другие ключи не добавляй. Значения — строки; если поля нет в тексте — \"\". "
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
        self.max_input_chars = max(1000, int(os.environ.get("EXTRACT_MAX_INPUT_CHARS", "6000")))
        self._max_new_tokens = max(64, int(os.environ.get("EXTRACT_MAX_NEW_TOKENS", "320")))

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
