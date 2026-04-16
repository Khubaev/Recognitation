# Qwen Fine-tuning (Minimal)

Minimal kit for SFT/LoRA fine-tuning of Qwen on invoice OCR templates.

## Files

- `train_sft.sample.jsonl` - sample dataset format (`messages`).
- `train_lora_qwen.py` - minimal LoRA training script.
- `merge_lora.py` - merges LoRA adapter into a full model directory.
- `requirements-finetune.txt` - optional dependencies for this flow.

## Quick Start

1) Create environment and install dependencies:

```bash
python3 -m venv ~/ft-env
source ~/ft-env/bin/activate
pip install -r finetune/requirements-finetune.txt
```

Для **`build_sft_from_invoices`** (чтение PDF/Excel из папки с разметкой) дополнительно установите зависимости из **корня репозитория** — в чистом `vllm-env` или только с `requirements-finetune.txt` их обычно нет:

```bash
pip install -r requirements.txt
```

Минимум для конвертера: `pypdf` или `pymupdf`, `pandas`, `openpyxl`, для старых `.xls` — `xlrd`.

2) Prepare your dataset:

- Copy `finetune/train_sft.sample.jsonl` to `finetune/train_sft.jsonl`.
- Replace sample rows with your OCR->JSON training pairs.
- Or build from labeled invoices automatically:

```bash
python finetune/build_sft_from_invoices.py \
  --input-dir data/labeled_files/СчетНаОплату \
  --out finetune/train_sft.jsonl \
  --out-eval finetune/eval_sft.jsonl \
  --eval-ratio 0.15 \
  --text-mode auto
```

By default the converter uses only files with sidecar `*.json` labels and
builds the dataset with `--prompt-style neural_extract` (matches the production
prompt in `neural_extract.py`). Add `--include-silver` to include unlabeled
files with heuristic targets.

3) Train LoRA:

```bash
python finetune/train_lora_qwen.py \
  --data finetune/train_sft.jsonl \
  --output finetune/qwen7b-invoice-lora \
  --model Qwen/Qwen2.5-7B-Instruct
```

Defaults: 5 epochs, lr=1e-4, grad-accum=16, LoRA r=32/alpha=64.

4) Merge LoRA into standalone model:

```bash
python finetune/merge_lora.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora finetune/qwen7b-invoice-lora \
  --output finetune/qwen7b-invoice-merged
```

5) Run with vLLM:

```bash
vllm serve finetune/qwen7b-invoice-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 4096
```

## Iteration 1 (sidecar only, train/eval, prod prompt)

Цель: только размеченные `*.json` рядом с документами, без silver-эвристик; промпт как в проде (`EXTRACT_PREFIX`); небольшой eval-сплит и замер по полям после обучения.

1) Собрать датасет (15% на eval, промпт как у vLLM в `neural_extract.py`):

```bash
python -m finetune.build_sft_from_invoices \
  --input-dir data/labeled_files/СчетНаОплату \
  --out finetune/train_sft.jsonl \
  --out-eval finetune/eval_sft.jsonl \
  --eval-ratio 0.15 \
  --prompt-style neural_extract \
  --text-mode auto
```

Без `--include-silver` берутся только файлы с sidecar JSON.

2) Обучить LoRA (при необходимости уменьшите `--max-seq-len` / увеличьте `--grad-accum` на 24GB):

```bash
python finetune/train_lora_qwen.py \
  --data finetune/train_sft.jsonl \
  --output finetune/qwen7b-invoice-lora \
  --model Qwen/Qwen2.5-7B-Instruct
```

Запускается с дефолтами: 5 эпох, lr=1e-4, grad-accum=16, LoRA r=32/alpha=64.

3) Слить адаптер и поднять vLLM (см. выше), затем оценить на holdout:

```bash
python -m finetune.eval_sft_vllm --eval-jsonl finetune/eval_sft.jsonl
```

Нужны `VLLM_OPENAI_BASE` и `VLLM_MODEL` (или флаги `--base-url` / `--model`). Скрипт печатает совпадения по пяти полям: номер/дата/ИНН поставщика/поставщик/итого.

## Notes

- vLLM is inference-only. Training happens via `transformers/trl/peft`.
- Keep output JSON strict and consistent with your production schema.
- Start from 50-100 hard real documents, then iterate.
