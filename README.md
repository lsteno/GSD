# GSD

Code generation model fine-tuning and evaluation. This repo contains experiments with parameter-efficient fine-tuning methods and benchmarking on competitive programming problems.

## What's Here

### GSD-finetune/
Four different parameter-efficient fine-tuning approaches for small code models (1.5B-3B params):
- **LoRA** - Low-rank adaptation in attention layers
- **QLoRA** - Quantized LoRA (4-bit) for memory-constrained setups
- **BitFit** - Train only bias terms
- **Prefix** - Learnable prompt tokens

Each method has its own folder with identical tooling: data prep, training scripts, NVML energy logging, and model merging. All use the same training parameters so you can compare methods directly.

### LiveCodeBench/
Evaluation harness for code generation models. Tests on fresh programming contest problems from LeetCode, AtCoder, and CodeForces (1055+ problems from May 2023 onwards). Evaluates code generation, self-repair, execution prediction, and test output prediction.

The `eval_simple/` subdirectory has streamlined scripts for evaluating locally fine-tuned models.

## Typical Workflow

1. Pick a fine-tuning method in `GSD-finetune/`
2. Download a base model (e.g., Qwen2.5-Coder-1.5B-Instruct)
3. Prepare training data from a dataset like OpenCodeInstruct
4. Train with energy logging enabled
5. Merge adapter weights back into the base model (LoRA/QLoRA)
6. Evaluate on LiveCodeBench using the eval scripts
7. Compare energy consumption and pass@k scores across methods

Each folder has its own README with detailed setup and commands.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- NVIDIA GPU (16-40GB depending on method and model size)
- Standard ML libraries: transformers, peft, datasets

Check the `requirements.txt` in each subdirectory.

## Notes

- All fine-tuning methods support offline mode once you've cached models and datasets
- QLoRA is the most memory-efficient (can train 3B models on 16GB GPUs)
- LoRA is the fastest and most widely supported
- Energy logging requires NVML (optional, training works without it)

See individual READMEs for details.
