#!/usr/bin/env python3
"""
Fetch a Hugging Face dataset, optionally subsample, and emit a JSON file with
`input`/`output` pairs ready for LoRA training.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare instruction dataset")
    parser.add_argument(
        "--dataset-id",
        default="livecodebench/code_generation_lite",
        help="Hugging Face dataset identifier (default: livecodebench/code_generation_lite)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--input-field",
        default="input",
        help="Field name to use as instruction text (default: input)",
    )
    parser.add_argument(
        "--output-field",
        default="output",
        help="Field name to use as target text (default: output)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Optional number of samples to retain",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for shuffling before sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(
        args.dataset_id,
        split=args.split,
    )

    if args.input_field not in ds.column_names or args.output_field not in ds.column_names:
        raise ValueError(
            f"Requested fields not found. Available columns: {ds.column_names}"
        )

    total = len(ds)
    if args.sample_count is not None and args.sample_count < total:
        ds = ds.shuffle(seed=args.seed).select(range(args.sample_count))

    converted: list[dict[str, str]] = []
    for row in ds:
        input_text = row.get(args.input_field)
        output_text = row.get(args.output_field)
        if not input_text or not output_text:
            continue
        converted.append({"input": input_text, "output": output_text})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(converted)} samples (requested {args.sample_count or 'all'}, dataset had {total}) to {args.output}"
    )


if __name__ == "__main__":
    main()
