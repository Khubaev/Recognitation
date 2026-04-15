from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into a standalone model.")
    parser.add_argument("--base-model", required=True, help="Base model id/path (same as training base).")
    parser.add_argument("--lora", required=True, help="Path to trained LoRA adapter directory.")
    parser.add_argument("--output", required=True, help="Output directory for merged model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, args.lora)
    merged = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model saved to: {args.output}")


if __name__ == "__main__":
    main()
