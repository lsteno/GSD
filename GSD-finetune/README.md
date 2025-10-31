# GSD Fine-tune

Parameter-efficient fine-tuning experiments for code generation models with GPU energy tracking. This repo compares four different fine-tuning approaches on small language models (1.5B–3B parameters) using the same data prep, training infrastructure, and evaluation pipeline.

## What's Inside

Each subdirectory implements a different fine-tuning method with identical tooling:

- **`lora_simple/`** – Low-Rank Adaptation: freeze the base model, add trainable rank decomposition matrices to attention layers
- **`qlora_simple/`** – Quantized LoRA: same as LoRA but loads the base model in 4-bit to fit larger models on smaller GPUs
- **`bitfit_simple/`** – Bias Fine-Tuning: freeze everything except bias terms (simplest possible parameter-efficient method)
- **`prefix_simple/`** – Prefix Tuning: prepend learnable virtual tokens to each layer's key/value cache

All methods share the same offline workflow, NVML energy logging, and early stopping logic so you can compare training cost, memory usage, and downstream accuracy head-to-head.

## Quick Start

Pick a method folder and follow the steps below. Commands here use `lora_simple/` as the example; swap in `qlora_simple`, `bitfit_simple`, or `prefix_simple` to try a different approach.

### 1. Set up the environment

```bash
cd lora_simple
python3 -m venv .venv_simple
source .venv_simple/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs runs data
```

Repeat this for each method you want to try. The dependencies are lightweight (transformers, torch, peft, datasets, nvidia-ml-py) and mostly overlap across folders.

### 2. Download the base model

```bash
python3 download_model.py \
  --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --cache-dir ./model_cache
```

This fetches weights and tokenizer from Hugging Face and stores them locally so training can run offline.

### 3. Prepare training data

The dataset helpers convert Hugging Face datasets into a simple `{input, output}` JSON format that all trainers expect.

**Balanced sampling** (recommended for contest-style tasks):

```bash
python3 prepare_balanced_data.py \
  --dataset-id nvidia/OpenCodeInstruct \
  --split train \
  --generic-count 2500 \
  --algorithmic-count 2500 \
  --output data/train_5k.json

python3 prepare_balanced_data.py \
  --dataset-id nvidia/OpenCodeInstruct \
  --split train \
  --generic-count 200 \
  --algorithmic-count 200 \
  --seed 123 \
  --output data/dev_400.json
```

The script ensures equal representation from generic coding tasks and algorithmic problems. You can adjust counts or use a different dataset—just make sure the output JSON follows the `[{input, output}, ...]` schema.

**Simple slicing** (alternative):

```bash
python3 prepare_data.py \
  --dataset-id nvidia/OpenCodeInstruct \
  --split "train[:5000]" \
  --output data/train_5k.json
```

### 4. Train with energy logging

```bash
python3 train.py \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset data/train_5k.json \
  --dev-dataset data/dev_400.json \
  --output-dir runs/my-experiment \
  --cache-dir ./model_cache \
  --energy-log runs/my-experiment/energy.json \
  --epochs 10
```

What happens:
1. NVML reads the GPU energy counter (start)
2. Model loads from cache with the appropriate PEFT adapter (LoRA, QLoRA, etc.)
3. Data is tokenized using the model's chat template
4. Hugging Face Trainer runs with early stopping (patience=3, min improvement=1%)
5. Best checkpoint is saved based on dev loss
6. NVML reads the energy counter again and logs total joules

The script prints trainable parameter counts so you can see how much you're actually updating. Typical numbers:
- **LoRA** (r=8): ~0.3% of parameters
- **QLoRA** (r=8, 4-bit): ~0.3% (but way less GPU memory)
- **BitFit**: ~0.1% (only biases)
- **Prefix** (64 tokens): ~0.2–0.5% depending on model size

To run on SLURM instead:

```bash
sbatch simple_train.slurm
```

Check `logs/simple_*.out` for job output.

### 5. Export merged weights

LoRA and QLoRA need an extra merge step to fold adapters back into the base model:

```bash
python3 merge.py \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --lora-ckpt runs/my-experiment \
  --output-dir runs/my-experiment-merged \
  --cache-dir ./model_cache
```

BitFit saves a standard checkpoint directly (no merge needed). Prefix tuning also skips this step—the adapter directory is self-contained.

### 6. Evaluate on LiveCodeBench

Point the evaluation harness at your merged model:

```bash
bash eval_simple/run_eval_simple.sh \
  --model MyModel \
  --local-path ~/GSD-finetune/lora_simple/runs/my-experiment-merged
```

Or submit via SLURM:

```bash
sbatch eval_simple/simple_eval.slurm \
  --model MyModel \
  --local-path ~/GSD-finetune/lora_simple/runs/my-experiment-merged
```

Results appear in `output/MyModel/` with pass@k scores for each problem.

## Comparing Methods

Each method offers different trade-offs:

| Method | Trainable % | Memory | Training Speed | Merge? | Notes |
|--------|-------------|--------|----------------|--------|-------|
| **LoRA** | ~0.3% | Medium | Fast | Yes | Best all-around choice, widely supported |
| **QLoRA** | ~0.3% | Low | Slow | Yes | 4-bit quantization fits bigger models on small GPUs |
| **BitFit** | ~0.1% | Medium | Fast | No | Simplest method, surprisingly effective |
| **Prefix** | ~0.2–0.5% | Medium | Medium | No | Prepends virtual tokens, needs HF runner for eval |

To compare energy and accuracy across methods:

1. Train each variant with the same dataset
2. Collect `energy.json` from each run
3. Evaluate all merged models on the same benchmark
4. Run a report script (if available) to visualize energy vs. pass@1

The repo enforces identical training parameters (batch size, learning rate, early stopping) so differences in final quality reflect the inductive bias of each method, not hyperparameter luck.

## File Structure

Every method folder has the same layout:

```
method_simple/
├── README.md              # method-specific docs
├── requirements.txt       # dependencies
├── download_model.py      # fetch model to cache
├── prepare_data.py        # slice & format HF dataset
├── prepare_balanced_data.py  # domain-balanced sampling
├── nvml_utils.py          # GPU energy counter wrapper
├── train.py               # main fine-tuning script
├── merge.py               # export full weights (LoRA/QLoRA only)
└── simple_train.slurm     # SLURM wrapper
```

Shared utilities (`nvml_utils.py`, data prep scripts) are duplicated across folders so each method is self-contained. If you tweak one, copy changes to the others manually or symlink them.

## Common Flags

Training scripts accept the same CLI arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--base-model` | Model name or local path | required |
| `--dataset` | Training JSON | required |
| `--dev-dataset` | Validation JSON | `None` |
| `--output-dir` | Checkpoint directory | required |
| `--cache-dir` | Local model cache | `None` (uses HF default) |
| `--energy-log` | NVML log path | `None` (skip logging) |
| `--epochs` | Max training epochs | 3–10 (varies by method) |
| `--learning-rate` | Optimizer LR | 2e-4 to 5e-5 |
| `--batch-size` | Per-device batch size | 2 |
| `--gradient-accumulation` | Accumulation steps | 8 |
| `--max-length` | Context window | 2048 |
| `--gradient-checkpointing` | Enable checkpointing | off |

Method-specific flags:

- **LoRA/QLoRA**: `--lora-r`, `--lora-alpha`, `--lora-dropout`
- **QLoRA**: `--quant-bits`, `--quant-type`, `--compute-dtype`
- **BitFit**: `--trainable-pattern` (can specify multiple)
- **Prefix**: `--num-virtual-tokens`, `--prefix-projection`

## Offline Workflow

All scripts respect HuggingFace offline mode. Set these before training or eval:

```bash
export HF_HOME=/path/to/cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Download the model and dataset once with internet access, then everything else runs from cache.

## Tips

- Start with **LoRA** if you're new to PEFT—it's fast, well-documented, and works everywhere
- Try **QLoRA** if you're memory-constrained (e.g., training a 7B model on a 16GB GPU)
- Use **BitFit** as a lightweight baseline to see how much LoRA's rank decomposition actually helps
- Experiment with **Prefix** if your task benefits from prompt-style conditioning (rare for code, but interesting)

Keep the same seed (`--seed` in data prep, fixed in training) when comparing methods to reduce noise.

## Evaluation Notes

- **LoRA/QLoRA/BitFit**: use the standard vLLM-based eval runner (fast, production-grade inference)
- **Prefix**: use the HF transformers runner (`eval_simple/run_eval_simple.sh` with `--multiprocess`) since vLLM doesn't support prefix adapters yet

Check each method's README for detailed evaluation commands and SLURM scripts.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- Transformers, PEFT, Datasets (see `requirements.txt` in each folder)
- NVML bindings for energy logging (optional, needs NVIDIA GPU)
- 16–40GB VRAM depending on method and model size

QLoRA can train 3B models on 16GB cards; the others need ~24GB for the same size.

## Troubleshooting

**Out of memory during training?**
- Enable `--gradient-checkpointing`
- Reduce `--batch-size` (increase `--gradient-accumulation` to keep effective batch size)
- Try QLoRA with `--quant-bits 4`

**Evaluation failing?**
- Make sure `HF_HOME` points to the cached base model
- Check that merged weights include `config.json` and `tokenizer.json`
- Prefix tuning needs `prefix_metadata.json` alongside the adapter

**NVML errors?**
- Omit `--energy-log` if you don't have NVIDIA GPUs or the py3nvml package isn't installed
- Training still runs normally, just without energy tracking

**Baseline regression (fine-tuned model worse than base)?**
- Use balanced data prep to avoid domain collapse
- Lower the learning rate or enable early stopping with `--dev-dataset`
- Check if the dataset quality matches the base model's instruction format

## Citation

If you use this code for research, please cite the underlying methods:

- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- **QLoRA**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)
- **BitFit**: Ben Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning" (ACL 2022)
- **Prefix**: Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts" (ACL 2021)

## License

Check individual files for license headers. Most training scripts are provided as-is for research purposes.

---

Questions or suggestions? Open an issue or PR. Keep the changes minimal—this repo is intentionally bare-bones so it's easy to fork and modify.
