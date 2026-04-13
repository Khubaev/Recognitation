"""
Дообучение seq2seq (Flan-T5): текст счёта → JSON с реквизитами.
Перед запуском: py -3 prepare_extract_dataset.py

Точность и «правдивость» ответов в первую очередь задаются качеством и объёмом разметки (PDF + JSON).
Длиннее обучение (больше --epochs) помогает лучше сойтись на данных; при малой выборке возможно
переобучение — смотрите eval_loss на валидации (--valid-ratio > 0).
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from document_classifier.neural_extract import EXTRACT_PREFIX

DEFAULT_MODEL = "google/flan-t5-small"


class ExtractSeqDataset(Dataset):
    def __init__(self, texts: list[str], targets: list[str], tokenizer, max_src: int, max_tgt: int):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = EXTRACT_PREFIX + self.texts[idx]
        enc = self.tokenizer(
            src,
            truncation=True,
            max_length=self.max_src,
        )
        lab = self.tokenizer(
            text_target=self.targets[idx],
            truncation=True,
            max_length=self.max_tgt,
        )
        enc["labels"] = lab["input_ids"]
        return enc


def load_pairs(path: Path) -> tuple[list[str], list[str]]:
    texts, targets = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            t = (o.get("text") or "").strip()
            tgt = (o.get("target") or "").strip()
            if t and tgt:
                texts.append(t)
                targets.append(tgt)
    return texts, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/extract_train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/invoice_extract"))
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Число эпох (по умолчанию 50; больше — дольше, при росте датасета часто полезнее)",
    )
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--max-tgt", type=int, default=384)
    ap.add_argument("--valid-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Нет {args.train}. Сначала: py -3 prepare_extract_dataset.py")

    texts, targets = load_pairs(args.train)
    if len(texts) < 1:
        raise SystemExit("Пустой датасет извлечения")

    rng = random.Random(args.seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_val = int(round(len(texts) * args.valid_ratio))
    n_val = min(n_val, len(texts) - 1)
    if len(texts) < 4 or n_val < 1:
        n_val = 0

    if n_val:
        val_i = set(idx[:n_val])
        tr_i = [i for i in idx if i not in val_i]
        train_texts = [texts[i] for i in tr_i]
        train_tgt = [targets[i] for i in tr_i]
        val_texts = [texts[i] for i in val_i]
        val_tgt = [targets[i] for i in val_i]
    else:
        train_texts, train_tgt = texts, targets
        val_texts, val_tgt = [], []

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_ds = ExtractSeqDataset(train_texts, train_tgt, tokenizer, args.max_src, args.max_tgt)
    eval_ds = (
        ExtractSeqDataset(val_texts, val_tgt, tokenizer, args.max_src, args.max_tgt)
        if val_texts
        else None
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    args.out.mkdir(parents=True, exist_ok=True)

    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_policy = "epoch" if eval_ds else "no"
    eval_kw = (
        {"eval_strategy": eval_policy}
        if "eval_strategy" in ta_params
        else {"evaluation_strategy": eval_policy}
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=args.max_tgt,
        report_to="none",
        **eval_kw,
    )

    tr_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    tr_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    if "tokenizer" in tr_sig:
        tr_kw["tokenizer"] = tokenizer
    elif "processing_class" in tr_sig:
        tr_kw["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**tr_kw)

    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _save_weights(dst: Path) -> None:
        trainer.save_model(str(dst))
        tokenizer.save_pretrained(str(dst))

    try:
        _save_weights(out)
        print(f"Модель извлечения сохранена: {out}")
    except Exception as e:
        # Windows ERROR_USER_MAPPED_FILE (1224): «файл с открытой сопоставленной секцией» —
        # часто веса уже открыты другим процессом (например api_trained.py).
        err_s = f"{e}"
        if "1224" in err_s or "SafetensorError" in type(e).__name__:
            alt = out.parent / f"{out.name}_saved"
            print(
                f"\n[внимание] Не удалось записать в {out}:\n  {type(e).__name__}: {e}\n"
                f"Остановите процессы, которые читают {out} (часто HTTP API), и повторите сохранение.\n"
                f"Пробую альтернативную папку без перезаписи старых файлов: {alt}\n"
            )
            alt.mkdir(parents=True, exist_ok=True)
            _save_weights(alt)
            print(f"Модель извлечения сохранена: {alt}")
            print(
                f"Скопируйте содержимое в {out} при остановленном API или задайте "
                f"INVOICE_EXTRACT_CKPT={alt.resolve()}"
            )
        else:
            raise


if __name__ == "__main__":
    main()
