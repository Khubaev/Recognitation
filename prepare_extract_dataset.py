"""
Сбор датасета для обучения извлечения полей из счетов на оплату.

1) Всегда подмешивает data/extract_seed.jsonl (ручные примеры).
2) Опционально: PDF/docx/xlsx/xls и изображения png/jpg/jpeg из папки (по умолчанию data/labeled_files/СчетНаОплату; для картинок нужен Yandex OCR).
   - Если рядом лежит одноимённый .json с эталоном полей — берётся он (плоский русский JSON или массив с {recipient, buyer, items}).
   - Иначе цель строится эвристикой extract_invoice_fields (серебряная разметка).
3) JSON без документа: если для стема нет PDF/xlsx/xls, но есть только .json с полем "text" и разметкой — строка попадёт в датасет.

Выход: data/extract_train.jsonl — строки {"text": "...", "target": "<json>"}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Как в api_trained: Yandex OCR из .env при сборе датасета с JPG/PNG
load_dotenv(Path(__file__).resolve().parent / ".env")

from document_classifier.extract import extract_text_from_file
from document_classifier.extract_target import fields_to_target_json
from document_classifier.invoice_fields import extract_invoice_fields
from document_classifier.labeled_data import SUPPORTED

SEED = Path("data/extract_seed.jsonl")
DEFAULT_INVOICES = Path("data/labeled_files/СчетНаОплату")
OUT = Path("data/extract_train.jsonl")


def row_from_record(rec: dict) -> tuple[str, str] | None:
    text = (rec.get("text") or "").strip()
    if not text:
        return None
    if "target" in rec and rec["target"]:
        return text, str(rec["target"]).strip()
    if "fields" in rec and isinstance(rec["fields"], dict):
        return text, fields_to_target_json(rec["fields"])
    return None


def load_sidecar_json(doc_path: Path) -> dict | None:
    p = doc_path.with_suffix(".json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--invoices-dir",
        type=Path,
        default=DEFAULT_INVOICES,
        help="Папка со счетами (PDF/xlsx/xls)",
    )
    ap.add_argument("--seed", type=Path, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows: list[tuple[str, str]] = []

    if args.seed.exists():
        with open(args.seed, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                got = row_from_record(r)
                if got:
                    rows.append(got)

    inv_dir = args.invoices_dir
    # При нескольких файлах с одним stem: сначала векторный PDF/Excel, потом сканы
    ext_order = {".pdf": 0, ".docx": 1, ".xlsx": 2, ".xls": 3, ".png": 4, ".jpg": 5, ".jpeg": 5}
    if inv_dir.is_dir():
        by_stem: dict[str, list[Path]] = defaultdict(list)
        for p in inv_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED:
                continue
            by_stem[p.stem.lower()].append(p)
        for stem in sorted(by_stem.keys()):
            plist = sorted(by_stem[stem], key=lambda x: ext_order.get(x.suffix.lower(), 9))
            p = plist[0]
            if len(plist) > 1:
                print(f"[один вариант из {len(plist)}] {p.name}")
            try:
                text = extract_text_from_file(p)
            except Exception as e:
                print(f"[пропуск] {p}: {e}")
                continue
            if not text.strip():
                print(f"[пусто] {p}")
                continue
            gold = load_sidecar_json(p)
            if gold:
                target = fields_to_target_json(gold)
            else:
                target = fields_to_target_json(extract_invoice_fields(text))
            rows.append((text, target))
            print(f"+ {p.name} ({len(text)} sym)")

        # JSON с встроенным "text", если нет файла с тем же stem (PDF/xlsx/xls)
        for jp in sorted(inv_dir.rglob("*.json")):
            if jp.name.startswith("_"):
                continue
            if jp.stem.lower() in by_stem:
                continue
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                continue
            txt = (data.get("text") or "").strip()
            if not txt:
                continue
            body = {k: v for k, v in data.items() if k != "text"}
            target = fields_to_target_json(body)
            rows.append((txt, target))
            print(f"+ {jp.name} (json+text, {len(txt)} sym)")

    if not rows:
        raise SystemExit(
            f"Нет примеров. Добавьте строки в {args.seed} и/или файлы в {inv_dir}",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for text, target in rows:
            f.write(json.dumps({"text": text, "target": target}, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
