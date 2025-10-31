#!/usr/bin/env python3
"""
Export a BitFit fine-tuned checkpoint as a standalone `transformers` model.
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export BitFit checkpoint")
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model name or path (used as a fallback for tokenizer files)",
    )
    parser.add_argument(
        "--bitfit-ckpt",
        required=True,
        type=Path,
        help="Directory produced by training (contains pytorch_model.bin)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the merged model",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.bitfit_ckpt,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    model.save_pretrained(args.output_dir, safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.bitfit_ckpt,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged BitFit model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
