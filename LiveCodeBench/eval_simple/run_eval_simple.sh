#!/bin/bash
# Minimal LiveCodeBench evaluation helper.
# Usage:
#   sbatch eval_simple/run_eval_simple.sh \
#     --model Qwen2.5-1.5B-QLoRA \
#     --local-path ~/GSD-finetune/qlora_simple/runs/qwen2.5-1.5b-qlora-merged

set -euo pipefail

MODEL=""
LOCAL_PATH=""
RELEASE="v6"
SCENARIO="codegeneration"
N=10
TEMPERATURE=0.2
REPETITIONS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --local-path) LOCAL_PATH="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --model NAME --local-path DIR [--release v6] [--scenario codegeneration] [--n 10] [--temperature 0.2] [--repetitions 1]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$LOCAL_PATH" ]]; then
  echo "Both --model and --local-path are required." >&2
  exit 1
fi

# Offline by default (override HF_HOME/HF_DATASETS_CACHE before calling if desired)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
source ~/LiveCodeBench/.venv/bin/activate

# Expand user in the provided local path and fail fast if it is missing
LOCAL_PATH=$(python -c 'import os,sys; print(os.path.expanduser(sys.argv[1]))' "$LOCAL_PATH")
if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "Local path '$LOCAL_PATH' does not exist or is not a directory." >&2
  exit 1
fi
  
# Create logs directory next to this script if it doesn't exist
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="../logs"

# Generate a base timestamp for this batch of runs
BASE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=${MODEL//\//_}
MODEL_SAFE=${MODEL_SAFE// /_}

echo "Running $REPETITIONS repetition(s) of the evaluation..."

# Loop through the number of repetitions
for ((i=1; i<=REPETITIONS; i++)); do
  echo ""
  echo "=========================================="
  echo "Starting repetition $i of $REPETITIONS"
  echo "=========================================="
  
  # Generate log and error filenames with timestamp, repetition number, and a filesystem-safe model name
  LOG_FILE="$LOG_DIR/${MODEL_SAFE}_${SCENARIO}_${RELEASE}_${BASE_TIMESTAMP}_rep${i}.log"
  ERR_FILE="$LOG_DIR/${MODEL_SAFE}_${SCENARIO}_${RELEASE}_${BASE_TIMESTAMP}_rep${i}.err"
  
  echo "Logging to: $LOG_FILE"
  echo "Errors to: $ERR_FILE"
  
  # Set environment variables for batch organization if running multiple repetitions
  if [ "$REPETITIONS" -gt 1 ]; then
    export LCB_OUTPUT_SUFFIX="_batch_${BASE_TIMESTAMP}/rep${i}"
    echo "Output directory: output/\${MODEL}${LCB_OUTPUT_SUFFIX}/"
  else
    unset LCB_OUTPUT_SUFFIX
    echo "Output directory: output/\${MODEL}/"
  fi
  
  # Run python with stdout to log file and stderr to err file
  # Using explicit redirection that works in Slurm non-interactive environments
  python -m lcb_runner.runner.main \
    --model "$MODEL" \
    --local_model_path "$LOCAL_PATH" \
    --scenario "$SCENARIO" \
    --evaluate \
    --release_version "$RELEASE" \
    --n "$N" \
    --temperature "$TEMPERATURE" \
    --tensor_parallel_size 2 \
    --peft_adapter_path "/home/s3221407/GSD-finetune/prefix_simple/runs/qwen2.5-1.5b-prefix-5k" \
    > >(tee "$LOG_FILE") \
    2> >(tee "$ERR_FILE" >&2)
  
  echo ""
  echo "Completed repetition $i of $REPETITIONS"
done

echo ""
echo "=========================================="
echo "All $REPETITIONS repetition(s) completed!"
echo "=========================================="
