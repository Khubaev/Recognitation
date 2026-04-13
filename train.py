"""
Дообучение классификатора типов первичных документов по тексту (после OCR или из XML/JSON).
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from document_classifier.config import DEFAULT_MODEL_NAME, DOC_LABELS, ID2LABEL, LABEL2ID
from document_classifier.dataset import load_jsonl, load_labeled_file
from document_classifier.labeled_data import load_samples_from_labeled_root, train_valid_split


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    ap = argparse.ArgumentParser(
        description="Обучение: jsonl и/или папки с PDF/Excel по меткам (см. --labeled-root).",
    )
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument(
        "--valid",
        type=Path,
        default=None,
        help="Файл валидации: .json (массив) или .jsonl. Если не указан и есть data/valid.json — он подставится.",
    )
    ap.add_argument(
        "--labeled-root",
        type=Path,
        default=None,
        help="Каталог с подпапками-метками (имя = класс), внутри PDF/xlsx/xls",
    )
    ap.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help="Доля валидации при обучении только с --labeled-root (если нет --valid)",
    )
    ap.add_argument("--out", type=Path, default=Path("checkpoints/doc_classifier"))
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Число эпох (при необходимости увеличьте для более длительного обучения)",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    valid_path = args.valid
    if valid_path is None:
        auto = Path("data/valid.json")
        if auto.exists():
            valid_path = auto

    train_rows: list = []
    eval_rows: list = []

    if args.labeled_root:
        labeled = load_samples_from_labeled_root(Path(args.labeled_root))
        explicit_valid = valid_path is not None and Path(valid_path).exists()
        if explicit_valid:
            train_rows.extend(labeled)
        elif args.valid_ratio > 0:
            tr, va = train_valid_split(labeled, args.valid_ratio)
            train_rows.extend(tr)
            eval_rows.extend(va)
        else:
            train_rows.extend(labeled)

    if args.train.exists():
        train_rows.extend(load_jsonl(args.train))

    if valid_path is not None and Path(valid_path).exists():
        eval_rows = load_labeled_file(Path(valid_path))

    if not train_rows:
        raise SystemExit(
            "Нет обучающих данных: задайте --labeled-root с файлами и/или существующий --train (jsonl).",
        )

    train_texts = [t for t, _ in train_rows]
    train_labels = [y for _, y in train_rows]

    eval_dataset = None
    if eval_rows:
        eval_texts = [t for t, _ in eval_rows]
        eval_labels = [y for _, y in eval_rows]
    else:
        eval_texts, eval_labels = [], []

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(DOC_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    if eval_texts:
        eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer, args.max_length)

    args.out.mkdir(parents=True, exist_ok=True)

    eval_policy = "epoch" if eval_dataset else "no"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_kw = (
        {"eval_strategy": eval_policy}
        if "eval_strategy" in ta_params
        else {"evaluation_strategy": eval_policy}
    )

    training_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=10,
        **eval_kw,
        save_strategy="epoch",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="accuracy" if eval_dataset else None,
        save_total_limit=1,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    tr_params = inspect.signature(Trainer.__init__).parameters
    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None,
    )
    if "tokenizer" in tr_params:
        trainer_kw["tokenizer"] = tokenizer
    elif "processing_class" in tr_params:
        trainer_kw["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kw)

    trainer.train()

    if eval_dataset:
        pred = trainer.predict(eval_dataset)
        y_true = pred.label_ids
        y_pred = pred.predictions.argmax(-1)
        print(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(len(DOC_LABELS))),
                target_names=DOC_LABELS,
                digits=3,
                zero_division=0,
            ),
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _save_ckpt(dst: Path) -> None:
        trainer.save_model(str(dst))
        tokenizer.save_pretrained(str(dst))

    try:
        _save_ckpt(out)
        print(f"Модель сохранена: {out}")
    except Exception as e:
        err_s = f"{e}"
        if "1224" in err_s or "SafetensorError" in type(e).__name__:
            alt = out.parent / f"{out.name}_saved"
            print(
                f"\n[внимание] Не удалось записать в {out}:\n  {type(e).__name__}: {e}\n"
                f"Остановите процессы, которые читают {out} (часто HTTP API), и повторите сохранение.\n"
                f"Пробую альтернативную папку: {alt}\n"
            )
            alt.mkdir(parents=True, exist_ok=True)
            _save_ckpt(alt)
            print(f"Модель сохранена: {alt}")
            print(
                f"Скопируйте содержимое в {out} при остановленном API или задайте "
                f"DOC_CLASSIFIER_CKPT={alt.resolve()}"
            )
        else:
            raise


if __name__ == "__main__":
    main()
