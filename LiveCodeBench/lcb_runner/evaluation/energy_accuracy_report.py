"""
Utility script to relate NVML energy measurements to LiveCodeBench accuracy gains.

Example
-------
python -m lcb_runner.evaluation.energy_accuracy_report \
    --base-eval output/Qwen2.5-Ins-0.5B/codegeneration_10_0.2_eval_all.json \
    --base-energy logs/energy_Qwen2.5-Ins-0.5B_402500.json \
    --finetune-eval output/Qwen2.5-0.5B-FT/codegeneration_10_0.2_eval_all.json \
    --finetune-energy logs/energy_Qwen2.5-0.5B-Finetuned_402589.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute energy-to-accuracy trade-offs for base vs finetuned runs."
    )
    parser.add_argument("--base-eval", required=True, help="Eval_all JSON for the base model")
    parser.add_argument("--base-energy", required=True, help="Energy summary JSON for the base run")
    parser.add_argument(
        "--finetune-eval", required=True, help="Eval_all JSON for the finetuned model"
    )
    parser.add_argument(
        "--finetune-energy", required=True, help="Energy summary JSON for the finetuned run"
    )
    parser.add_argument(
        "--ks",
        default="1,5",
        help="Comma separated list of Pass@k values to analyse (default: 1,5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the summary report as JSON",
    )
    return parser.parse_args()


def load_energy(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if "energy_joules" not in data:
        raise ValueError(f"{path} missing 'energy_joules'")
    return data


def compute_pass_metrics(eval_path: str | Path, ks: Iterable[int]) -> dict[str, float]:
    with open(eval_path, "r", encoding="utf-8") as fp:
        results = json.load(fp)
    if not results:
        raise ValueError(f"{eval_path} contains no evaluation results")
    totals = [len(entry["graded_list"]) for entry in results]
    corrects = [sum(entry["graded_list"]) for entry in results]
    metrics: dict[str, float] = {}
    for k in ks:
        pass_estimate = estimate_pass_at_k(totals, corrects, k)
        metrics[f"pass@{k}"] = float(np.mean(pass_estimate))
    return metrics


def build_report(base_eval_path: str, base_energy_path: str, finetune_eval_path: str, finetune_energy_path: str, ks: list[int]) -> dict:
    base_energy = load_energy(base_energy_path)
    finetune_energy = load_energy(finetune_energy_path)

    base_metrics = compute_pass_metrics(base_eval_path, ks)
    finetune_metrics = compute_pass_metrics(finetune_eval_path, ks)

    report = {
        "base": {
            "energy_joules": float(base_energy["energy_joules"]),
            **{k: base_metrics[k] for k in sorted(base_metrics)},
        },
        "finetune": {
            "energy_joules": float(finetune_energy["energy_joules"]),
            **{k: finetune_metrics[k] for k in sorted(finetune_metrics)},
        },
    }

    delta_energy = report["finetune"]["energy_joules"] - report["base"]["energy_joules"]
    report["delta"] = {"energy_joules": delta_energy}

    energy_per_gain: dict[str, float | None] = {}
    for k in sorted(base_metrics):
        gain = report["finetune"][k] - report["base"][k]
        report["delta"][k] = gain
        if gain > 0:
            energy_per_gain[k] = delta_energy / gain
        else:
            energy_per_gain[k] = None
    report["energy_per_gain"] = energy_per_gain
    return report


def main() -> None:
    args = parse_args()
    ks = [int(item.strip()) for item in args.ks.split(",") if item.strip()]
    report = build_report(
        args.base_eval, args.base_energy, args.finetune_eval, args.finetune_energy, ks
    )

    print(json.dumps(report, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)


if __name__ == "__main__":
    main()
