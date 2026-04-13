"""
Размеченные файлы: одна папка на класс, внутри — PDF / Word / Excel / изображения.
Текст из Word/Excel — локально; PDF и сканы — см. extract_text_from_file и OCR_YANDEX_*.
Имя подпапки должно совпадать с меткой из config.DOC_LABELS.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from tqdm import tqdm

from .config import LABEL2ID
from .extract import extract_text_from_file

SUPPORTED = {".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}


def iter_labeled_files(root: Path) -> Iterator[Tuple[Path, str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Нет каталога: {root}")

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label.startswith("."):
            continue
        if label not in LABEL2ID:
            tqdm.write(f"[внимание] папка не совпадает с меткой из config, пропуск: {label}")
            continue
        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                yield p, label


def load_samples_from_labeled_root(
    root: Path,
    *,
    show_progress: bool = True,
    skip_empty: bool = True,
) -> List[Tuple[str, int]]:
    """Возвращает список (текст, label_id). Пустой текст пропускается при skip_empty."""
    samples: List[Tuple[str, int]] = []
    paths_labels = list(iter_labeled_files(root))
    it = tqdm(paths_labels, desc="Извлечение текста") if show_progress else paths_labels

    for path, label_name in it:
        try:
            text = extract_text_from_file(path)
        except Exception as e:
            tqdm.write(f"[пропуск] {path}: {e}")
            continue
        if skip_empty and not text.strip():
            tqdm.write(f"[пусто] {path} — нет текстового слоя (скан?)")
            continue
        samples.append((text, LABEL2ID[label_name]))

    return samples


def train_valid_split(
    samples: Sequence[Tuple[str, int]],
    valid_ratio: float,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if valid_ratio <= 0:
        return list(samples), []
    if valid_ratio >= 1:
        raise ValueError("valid_ratio должен быть < 1")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    if n == 1:
        return shuffled, []

    n_val = max(1, int(round(n * valid_ratio)))
    n_val = min(n_val, n - 1)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
