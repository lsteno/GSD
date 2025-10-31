#!/usr/bin/env python3


import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare balanced instruction dataset by domain")
    parser.add_argument(
        "--dataset-id",
        default="nvidia/OpenCodeInstruct",
        help="Hugging Face dataset identifier (default: nvidia/OpenCodeInstruct)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
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
        "--domain-field",
        default="domain",
        help="Field name for domain filtering (default: domain)",
    )
    parser.add_argument(
        "--generic-count",
        type=int,
        default=2500,
        help="Number of samples from 'generic' domain (default: 2500)",
    )
    parser.add_argument(
        "--algorithmic-count",
        type=int,
        default=2500,
        help="Number of samples from 'algorithmic' domain (default: 2500)",
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
    
    print(f"Loading dataset {args.dataset_id} (split: {args.split})...")
    ds = load_dataset(args.dataset_id, split=args.split)
    
    # Validate required fields
    required_fields = [args.input_field, args.output_field, args.domain_field]
    missing_fields = [f for f in required_fields if f not in ds.column_names]
    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}. "
            f"Available columns: {ds.column_names}"
        )
    
    print(f"Total samples in dataset: {len(ds)}")
    
    # Filter by domain
    print("\nFiltering by domain...")
    generic_ds = ds.filter(lambda x: x[args.domain_field] == "generic")
    algorithmic_ds = ds.filter(lambda x: x[args.domain_field] == "algorithmic")
    
    print(f"  Generic domain: {len(generic_ds)} samples available")
    print(f"  Algorithmic domain: {len(algorithmic_ds)} samples available")
    
    # Check if we have enough samples
    if len(generic_ds) < args.generic_count:
        print(f"  WARNING: Requested {args.generic_count} generic samples, but only {len(generic_ds)} available")
        args.generic_count = len(generic_ds)
    
    if len(algorithmic_ds) < args.algorithmic_count:
        print(f"  WARNING: Requested {args.algorithmic_count} algorithmic samples, but only {len(algorithmic_ds)} available")
        args.algorithmic_count = len(algorithmic_ds)
    
    # Sample from each domain
    print(f"\nSampling {args.generic_count} generic + {args.algorithmic_count} algorithmic samples...")
    generic_sampled = generic_ds.shuffle(seed=args.seed).select(range(args.generic_count))
    algorithmic_sampled = algorithmic_ds.shuffle(seed=args.seed).select(range(args.algorithmic_count))
    
    # Convert to our format
    converted: list[dict[str, str]] = []
    
    for row in generic_sampled:
        input_text = row.get(args.input_field)
        output_text = row.get(args.output_field)
        if input_text and output_text:
            converted.append({"input": input_text, "output": output_text})
    
    for row in algorithmic_sampled:
        input_text = row.get(args.input_field)
        output_text = row.get(args.output_field)
        if input_text and output_text:
            converted.append({"input": input_text, "output": output_text})
    
    # Shuffle the combined dataset
    import random
    random.seed(args.seed)
    random.shuffle(converted)
    
    # Save to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    
    print(f"\n✓ Wrote {len(converted)} samples to {args.output}")
    print(f"  ({args.generic_count} generic + {args.algorithmic_count} algorithmic)")


if __name__ == "__main__":
    main()
