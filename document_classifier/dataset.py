import json
from pathlib import Path
from typing import List, Tuple

from .config import LABEL2ID


def _rows_from_list(objs: list) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for obj in objs:
        if not isinstance(obj, dict):
            raise ValueError("Каждый элемент должен быть объектом с полями text и label")
        text = (obj.get("text") or "").strip()
        label_name = obj.get("label")
        if not text or label_name is None:
            continue
        if label_name not in LABEL2ID:
            raise ValueError(f"Неизвестная метка: {label_name}")
        rows.append((text, LABEL2ID[label_name]))
    return rows


def load_labeled_json(path: Path) -> List[Tuple[str, int]]:
    """
    Один JSON-файл: массив объектов [{\"text\": \"...\", \"label\": \"СчетНаОплату\"}, ...]
    или {\"samples\": [ ... ]}.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "samples" in raw:
            raw = raw["samples"]
        elif "data" in raw:
            raw = raw["data"]
        else:
            raise ValueError("JSON: нужен массив или объект с ключом samples / data")
    if not isinstance(raw, list):
        raise ValueError("JSON: ожидается список примеров")
    return _rows_from_list(raw)


def load_labeled_file(path: Path) -> List[Tuple[str, int]]:
    """jsonl (по строкам) или .json (массив)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return load_jsonl(path)
    if suf == ".json":
        return load_labeled_json(path)
    raise ValueError(f"Неподдерживаемый формат валидации: {path}")


def load_jsonl(path: Path) -> List[Tuple[str, int]]:
    """Формат строки: {\"text\": \"...\", \"label\": \"СчетНаОплату\"}"""
    rows: List[Tuple[str, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"].strip()
            label_name = obj["label"]
            if label_name not in LABEL2ID:
                raise ValueError(f"Неизвестная метка: {label_name}")
            rows.append((text, LABEL2ID[label_name]))
    return rows


def save_jsonl(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, label_name in rows:
            f.write(json.dumps({"text": text, "label": label_name}, ensure_ascii=False) + "\n")
