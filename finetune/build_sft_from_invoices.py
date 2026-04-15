from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from document_classifier.extract import extract_text_from_file
from document_classifier.extract_target import canonicalize_extract_labels
from document_classifier.invoice_fields import extract_invoice_fields
from document_classifier.labeled_data import SUPPORTED

DEFAULT_INPUT_DIR = Path("data/labeled_files/СчетНаОплату")
DEFAULT_OUTPUT = Path("finetune/train_sft.jsonl")
DEFAULT_TEXT_MODE = "auto"
DEFAULT_SYSTEM = (
    "Ты извлекаешь данные из OCR-текста счета на оплату. "
    "Верни строго один JSON-объект с ключами: "
    "Номер счета, Дата счета, ИНН поставщика, Поставщик, Итого. "
    "Если значение отсутствует или нераспознано уверенно — null. "
    "Без markdown и пояснений."
)


def _load_sidecar_json(doc_path: Path) -> Optional[Dict[str, Any]]:
    p = doc_path.with_suffix(".json")
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    return raw if isinstance(raw, dict) else None


def _iso_date_or_none(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    # dd.mm.yyyy / dd/mm/yyyy
    import re

    m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$", s)
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if len(y) == 2:
        yi = int(y)
        y = f"20{yi:02d}" if yi < 70 else f"19{yi:02d}"
    return f"{int(y):04d}-{mo:02d}-{d:02d}"


def _to_number_or_none(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    cleaned = s.replace("\xa0", " ").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_target_fields(raw_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит любое sidecar/эвристику к минимальному target-JSON из 5 полей.
    """
    can = canonicalize_extract_labels(raw_fields or {})
    inv = (can.get("Номер счета") or "").strip()
    dt = (can.get("Дата счета") or "").strip()
    inn = (can.get("ИНН поставщика") or "").strip()
    supplier = (can.get("Поставщик") or "").strip()
    total = (can.get("Итого") or "").strip()
    return {
        "Номер счета": inv or None,
        "Дата счета": _iso_date_or_none(dt),
        "ИНН поставщика": inn or None,
        "Поставщик": supplier or None,
        "Итого": _to_number_or_none(total),
    }


def _make_messages(
    system_prompt: str,
    text: str,
    target_obj: Dict[str, Any],
    *,
    prompt_style: str,
) -> Dict[str, Any]:
    """prompt_style=sft_default — system + OCR; neural_extract — как в prod (EXTRACT_PREFIX + текст, один user)."""
    target_json = json.dumps(target_obj, ensure_ascii=False, separators=(",", ":"))
    if prompt_style == "neural_extract":
        from document_classifier.neural_extract import EXTRACT_PREFIX

        user_content = EXTRACT_PREFIX + text
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_json},
            ],
        }
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"OCR:\n{text}"},
            {"role": "assistant", "content": target_json},
        ],
    }


def _iter_invoice_docs(root: Path) -> Iterable[Path]:
    ext_order = {".pdf": 0, ".docx": 1, ".xlsx": 2, ".xls": 3, ".png": 4, ".jpg": 5, ".jpeg": 5}
    by_stem: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        by_stem.setdefault(p.stem.lower(), []).append(p)
    for stem in sorted(by_stem.keys()):
        plist = sorted(by_stem[stem], key=lambda x: ext_order.get(x.suffix.lower(), 9))
        yield plist[0]


def _build_sample(
    doc_path: Path,
    *,
    text_mode: str,
    system_prompt: str,
    include_silver: bool,
    prompt_style: str,
) -> Optional[Tuple[Dict[str, Any], str]]:
    text = extract_text_from_file(doc_path, text_extract_mode=text_mode)
    if not text.strip():
        return None
    sidecar = _load_sidecar_json(doc_path)
    if sidecar is None and not include_silver:
        return None
    target_raw = sidecar if sidecar is not None else extract_invoice_fields(text)
    target = _normalize_target_fields(target_raw)
    sample = _make_messages(system_prompt, text, target, prompt_style=prompt_style)
    source = "sidecar" if sidecar is not None else "silver"
    return sample, source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT jsonl for Qwen from labeled invoice files.")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder with invoice files.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output jsonl path.")
    p.add_argument(
        "--text-mode",
        choices=("auto", "local", "ocr"),
        default=DEFAULT_TEXT_MODE,
        help="Text extraction mode (same as API).",
    )
    p.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM, help="System prompt for SFT messages.")
    p.add_argument(
        "--include-silver",
        action="store_true",
        help="Include files without sidecar json (target from regex heuristics).",
    )
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=0.0,
        help="If >0, hold out this fraction of samples for eval (deterministic shuffle, seed 42).",
    )
    p.add_argument(
        "--out-eval",
        type=Path,
        default=Path("finetune/eval_sft.jsonl"),
        help="Path for eval split when --eval-ratio > 0.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for train/eval split.")
    p.add_argument(
        "--prompt-style",
        choices=("sft_default", "neural_extract"),
        default="sft_default",
        help="neural_extract: как в prod (EXTRACT_PREFIX + текст). sft_default: system + OCR.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    rows: List[Dict[str, Any]] = []
    used_sidecar = 0
    used_silver = 0
    skipped = 0

    for p in _iter_invoice_docs(args.input_dir):
        try:
            built = _build_sample(
                p,
                text_mode=args.text_mode,
                system_prompt=args.system_prompt,
                include_silver=args.include_silver,
                prompt_style=args.prompt_style,
            )
        except Exception as e:
            skipped += 1
            print(f"[skip] {p.name}: {e}")
            continue
        if built is None:
            skipped += 1
            print(f"[skip] {p.name}: no sidecar json")
            continue
        sample, src = built
        rows.append(sample)
        if src == "sidecar":
            used_sidecar += 1
        else:
            used_silver += 1
        print(f"+ {p.name} ({src})")

    if not rows:
        raise SystemExit(
            "No samples built. Add sidecar .json labels or use --include-silver for heuristic targets.",
        )

    eval_ratio = max(0.0, min(0.5, float(args.eval_ratio)))
    train_rows = rows
    eval_rows: List[Dict[str, Any]] = []
    if eval_ratio > 0 and len(rows) >= 2:
        rng = random.Random(args.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_eval = max(1, int(round(len(rows) * eval_ratio)))
        n_eval = min(n_eval, len(rows) - 1)
        eval_set = set(idx[:n_eval])
        train_rows = [rows[i] for i in range(len(rows)) if i not in eval_set]
        eval_rows = [rows[i] for i in range(len(rows)) if i in eval_set]
        print(f"Split: train={len(train_rows)}, eval={len(eval_rows)} (ratio≈{eval_ratio})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if eval_rows:
        args.out_eval.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_eval, "w", encoding="utf-8") as f:
            for row in eval_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Eval saved to {args.out_eval}")

    print(
        f"Saved {len(train_rows)} train samples to {args.out} "
        f"(sidecar={used_sidecar}, silver={used_silver}, skipped={skipped})",
    )


if __name__ == "__main__":
    main()
