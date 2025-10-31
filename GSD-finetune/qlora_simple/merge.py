#!/usr/bin/env python3
"""
Merge QLoRA adapters into the base model to obtain standalone weights.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge QLoRA adapters")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--lora-ckpt", required=True, type=Path, help="LoRA checkpoint dir")
    parser.add_argument("--output-dir", required=True, type=Path, help="Merged output dir")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    adapter_config = args.lora_ckpt / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"QLoRA adapter checkpoint not found at {args.lora_ckpt}. "
            "Make sure training finished and the path contains adapter_config.json."
        )

    lora = PeftModel.from_pretrained(
        base,
        str(args.lora_ckpt),
        torch_dtype=torch.bfloat16,
    )
    merged = lora.merge_and_unload()

    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
