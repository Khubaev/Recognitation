"""Создаёт data/labeled_files/<метка>/.gitkeep для всех классов из config."""
from pathlib import Path

from document_classifier.config import DOC_LABELS

ROOT = Path("data/labeled_files")


def main():
    for name in DOC_LABELS:
        d = ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("", encoding="utf-8")
    print(f"Готово: {ROOT} — положите PDF/xlsx/xls в соответствующие подпапки.")


if __name__ == "__main__":
    main()
