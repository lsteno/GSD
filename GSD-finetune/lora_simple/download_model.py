#!/usr/bin/env python3
"""
Download a base model and tokenizer into a local cache directory.
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Download base model for offline use")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./model_cache"),
        help="Directory to store model files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tokenizer for {args.model_name}...")
    AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    print("Tokenizer done.")

    print("Downloading model weights (this may take a while)...")
    AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    print(f"Model cached under {args.cache_dir}")


if __name__ == "__main__":
    main()
