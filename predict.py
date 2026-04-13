"""
Инференс: тип документа и реквизиты.
По умолчанию: нейросеть + эвристики для незаполненных полей; --merge-fields явно то же самое.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_classifier.extract import resolve_readable_document
from document_classifier.inference import DocumentClassifier
from document_classifier.invoice_fields import FIELD_LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/doc_classifier"))
    ap.add_argument("--text", type=str, default=None, help="Текст или первые строки документа")
    ap.add_argument("--file", type=Path, default=None, help="UTF-8/csv текстовый файл")
    ap.add_argument(
        "--document",
        type=Path,
        default=None,
        help="Путь к .pdf / .xlsx / .xls (или к папке — возьмётся первый PDF/xlsx/xls)",
    )
    ap.add_argument(
        "--no-neural-fields",
        action="store_true",
        help="Не загружать модель извлечения; только эвристики",
    )
    ap.add_argument(
        "--merge-fields",
        action="store_true",
        help="Нейросеть + эвристики (заполнение полей, где нейросеть пустая)",
    )
    ap.add_argument(
        "--regex-only-fields",
        action="store_true",
        help="Только эвристики для полей",
    )
    ap.add_argument("--fields", action="store_true", help="Печать извлечённых реквизитов")
    ap.add_argument(
        "--text-extract-mode",
        choices=("auto", "local", "ocr"),
        default="auto",
        help="Как извлекать текст из файла до моделей (см. API text_extract_mode)",
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Нет чекпоинта: {args.checkpoint}. Сначала запустите train.py")

    if args.merge_fields and args.regex_only_fields:
        raise SystemExit("Нельзя одновременно --merge-fields и --regex-only-fields")

    if args.regex_only_fields:
        fields_mode = "regex_only"
    elif args.merge_fields:
        fields_mode = "merge"
    else:
        fields_mode = None

    clf = DocumentClassifier(
        args.checkpoint,
        use_neural_extract=not args.no_neural_fields,
        fields_mode="regex_only" if args.no_neural_fields else fields_mode,
    )

    if args.document:
        orig = Path(args.document).expanduser()
        try:
            doc_path = resolve_readable_document(orig)
        except (OSError, ValueError) as e:
            raise SystemExit(str(e)) from e
        if orig.is_dir():
            print(f"Используется файл из папки: {doc_path}")
        r = clf.predict_file(doc_path, text_extract_mode=args.text_extract_mode)
    elif args.file:
        text = args.file.read_text(encoding="utf-8", errors="replace")
        r = clf.predict_text(text)
    elif args.text:
        r = clf.predict_text(args.text)
    else:
        raise SystemExit("Укажите --text, --file или --document")

    if r.get("error") and args.document:
        print(f"Ошибка: {r['error']}")

    print(f"Тип: {r.get('label', '')} (p={r.get('confidence', 0):.4f})")
    for name, p in (r.get("top") or [])[:5]:
        print(f"  {name:<24} {p:.4f}")

    if args.fields:
        ext = r.get("fields") or {}
        print("\n--- Реквизиты ---")
        for k in FIELD_LABELS:
            print(f"{k}: {ext.get(k, '')}")
        if ext.get("items"):
            print("\nПозиции:")
            print(json.dumps(ext["items"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
