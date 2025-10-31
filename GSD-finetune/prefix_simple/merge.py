#!/usr/bin/env python3
"""
Prefix tuning does not require merging. This helper simply copies the adapter
directory (and optional tokenizer files) to a new destination for convenience.
"""

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy prefix-tuning checkpoint")
    parser.add_argument(
        "--prefix-ckpt",
        required=True,
        type=Path,
        help="Directory produced by prefix training (contains adapter_config.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory (must not exist)",
    )
    parser.add_argument(
        "--include-tokenizer",
        action="store_true",
        help="Copy tokenizer files alongside the adapter if present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists; remove it first.")

    adapter_config = args.prefix_ckpt / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"{args.prefix_ckpt} does not look like a PEFT adapter directory "
            "(missing adapter_config.json)."
        )

    shutil.copytree(args.prefix_ckpt, args.output_dir)

    if args.include_tokenizer:
        tokenizer_src = args.prefix_ckpt / "tokenizer_config.json"
        if tokenizer_src.exists():
            print("Tokenizer already bundled with adapter; nothing extra to copy.")
        else:
            print(
                "Tokenizer files not found in adapter directory. "
                "Copy them manually from the base model cache if needed."
            )

    print(f"Copied prefix checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
