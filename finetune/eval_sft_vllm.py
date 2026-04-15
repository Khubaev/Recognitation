"""
Оценка по eval jsonl (формат messages): запрос к vLLM OpenAI API, сравнение с эталоном assistant.

Использование (из корня репозитория):
  python -m finetune.eval_sft_vllm --eval-jsonl finetune/eval_sft.jsonl

Переменные окружения: VLLM_OPENAI_BASE, VLLM_MODEL, опционально VLLM_API_KEY, VLLM_TIMEOUT_SEC.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from document_classifier.extract_target import normalize_parsed, parse_model_json

KEYS: Tuple[str, ...] = (
    "Номер счета",
    "Дата счета",
    "ИНН поставщика",
    "Поставщик",
    "Итого",
)


def _messages_for_inference(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return []
    if messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def _load_gold(assistant_content: str) -> Dict[str, Any]:
    raw = json.loads(assistant_content)
    if not isinstance(raw, dict):
        return {}
    return raw


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _norm_num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace("\xa0", " ").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _field_match(key: str, gold: Any, pred: Any) -> bool:
    if key == "Итого":
        g = _norm_num(gold)
        p = _norm_num(pred)
        if g is None and p is None:
            return True
        if g is None or p is None:
            return False
        return abs(g - p) <= 0.02 + 1e-9 * max(abs(g), abs(p))
    return _norm_str(gold).lower() == _norm_str(pred).lower()


def _score_pair(gold: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[int, int]:
    ok = 0
    for k in KEYS:
        if _field_match(k, gold.get(k), pred.get(k)):
            ok += 1
    return ok, len(KEYS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval SFT eval jsonl against vLLM chat completions.")
    p.add_argument("--eval-jsonl", type=str, default="finetune/eval_sft.jsonl")
    p.add_argument("--base-url", type=str, default="", help="Override VLLM_OPENAI_BASE (e.g. http://127.0.0.1:8000/v1).")
    p.add_argument("--model", type=str, default="", help="Override VLLM_MODEL.")
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = (args.base_url or os.environ.get("VLLM_OPENAI_BASE") or "").strip().rstrip("/")
    model = (args.model or os.environ.get("VLLM_MODEL") or "").strip()
    if not base or not model:
        raise SystemExit("Задайте --base-url и --model или VLLM_OPENAI_BASE и VLLM_MODEL.")
    api_key = (os.environ.get("VLLM_API_KEY") or "").strip()
    timeout = float(os.environ.get("VLLM_TIMEOUT_SEC", "120"))

    path = args.eval_jsonl
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    total_ok = 0
    total_fields = 0
    n_rows = 0

    for i, line in enumerate(lines):
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            print(f"[skip {i}] bad messages")
            continue
        gold_text = messages[-1].get("content") if messages[-1].get("role") == "assistant" else ""
        if not isinstance(gold_text, str) or not gold_text.strip():
            print(f"[skip {i}] no assistant gold")
            continue
        try:
            gold = _load_gold(gold_text)
        except json.JSONDecodeError:
            print(f"[skip {i}] invalid gold JSON")
            continue

        infer_msgs = _messages_for_inference(messages)
        url = f"{base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": infer_msgs,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            print(f"[fail {i}] empty choices")
            continue
        raw = (choices[0].get("message") or {}).get("content") or ""
        if not isinstance(raw, str):
            raw = ""
        pred = normalize_parsed(parse_model_json(raw))

        ok, nf = _score_pair(gold, pred)
        total_ok += ok
        total_fields += nf
        n_rows += 1
        print(f"[{i}] fields {ok}/{nf}  pred={json.dumps(pred, ensure_ascii=False)[:200]}")

    if n_rows == 0:
        print("No rows evaluated.")
        return
    print(f"Done: {n_rows} rows, exact field accuracy {total_ok}/{total_fields} = {total_ok/total_fields:.4f}")


if __name__ == "__main__":
    main()
