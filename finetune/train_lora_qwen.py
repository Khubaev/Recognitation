from __future__ import annotations

import argparse
import inspect

from datasets import load_dataset
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal LoRA fine-tuning for Qwen invoice extraction.")
    parser.add_argument("--data", default="finetune/train_sft.jsonl", help="Path to training jsonl (messages format).")
    parser.add_argument("--output", default="finetune/qwen7b-invoice-lora", help="Output directory for LoRA adapter.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model id/path.")
    parser.add_argument("--epochs", type=float, default=5.0, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum training sequence length.")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit base model loading (not recommended on 24GB GPUs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=args.data, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.no_4bit:
        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    train_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": train_args,
    }
    # Compatibility across TRL versions: some versions dropped max_seq_length in SFTTrainer.__init__.
    sig = inspect.signature(SFTTrainer.__init__)
    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = args.max_seq_len
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to: {args.output}")


if __name__ == "__main__":
    main()
