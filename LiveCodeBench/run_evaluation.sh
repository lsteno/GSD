#!/bin/bash
#SBATCH --job-name=lcb-eval-qwen2.5-7b
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=students
#SBATCH --time=02:00:00
#SBATCH --mem=64GB

# IMPORTANT: Before submitting this job, make sure you've pre-downloaded the dataset
# on the head node (which has internet access). Download with:
#   python3 -c "from datasets import load_dataset; load_dataset('livecodebench/code_generation_lite', split='test', version_tag='v6', download_mode='force_redownload')"
# The dataset must be fully cached in ~/.cache/huggingface/datasets/ 
# This job runs in OFFLINE MODE - it cannot access the internet.

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load required modules
echo "Loading Python and CUDA modules..."
module load python/3.10.7
module load nvidia/cuda-11.8

# Activate virtual environment using absolute path
VENV_PATH="$(pwd)/.venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtual environment at: $VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first"
    exit 1
fi

# Verify Python is available
which python
python --version

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Set HuggingFace cache directory (datasets will be cached here)
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets/"

# CRITICAL: Force offline mode - no internet access on compute nodes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p logs

# Default values (can be overridden with CLI args)
MODEL_NAME="Qwen2.5-3B-Finetuned"
LOCAL_MODEL_PATH="~/GSD-finetune/model_cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
SCENARIO="codegeneration"
RELEASE_VERSION="v6"
N=10
TEMPERATURE=0.2
TENSOR_PARALLEL_SIZE=1
NUM_PROCESS_EVALUATE=12
TIMEOUT=10
DTYPE="bfloat16"


# Simple CLI parsing (accepts long options)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_NAME="$2"; shift 2;;
        --local-model-path)
            LOCAL_MODEL_PATH="$2"; shift 2;;
        --size-suffix)
            # convenience: append a size suffix to the model name and adjust path if desired
            SUFFIX="$2"
            MODEL_NAME="${MODEL_NAME%%-*}${SUFFIX}${MODEL_NAME#*-}"
            shift 2;;
        --n)
            N="$2"; shift 2;;
        --temperature)
            TEMPERATURE="$2"; shift 2;;
        --help|-h)
            usage;;
        *)
            echo "Unknown option: $1"; usage;;
    esac
done

echo "Starting LiveCodeBench evaluation..."
echo "Model name: $MODEL_NAME"
echo "Model path: $LOCAL_MODEL_PATH"
echo "Scenario: $SCENARIO"
echo "Release version: $RELEASE_VERSION"

# Expand tilde in LOCAL_MODEL_PATH if present
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH/#\~/$HOME}"

# Validate local model path exists
if [ ! -e "$LOCAL_MODEL_PATH" ]; then
    echo "Error: local model path does not exist: $LOCAL_MODEL_PATH" >&2
    echo "If your model is stored under ~/GSD-finetune/lora/, pass --local-model-path to point to the model directory." >&2
    exit 2
fi

if [ ! -r "$LOCAL_MODEL_PATH" ]; then
    echo "Error: local model path is not readable: $LOCAL_MODEL_PATH" >&2
    exit 3
fi

python -m lcb_runner.runner.main \
    --model "$MODEL_NAME" \
    --local_model_path "$LOCAL_MODEL_PATH" \
    --scenario "$SCENARIO" \
    --evaluate \
    --release_version "$RELEASE_VERSION" \
    --n "$N" \
    --temperature "$TEMPERATURE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_cache \
    --num_process_evaluate "$NUM_PROCESS_EVALUATE" \
    --timeout "$TIMEOUT" \
    --dtype "$DTYPE"

echo "Evaluation completed at: $(date)"

# Print location of results (best-effort guess based on model name)
OUT_DIR_NAME=$(echo "$MODEL_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')
echo ""
echo "Results saved in (approx):"
echo "  - output/${OUT_DIR_NAME}/${SCENARIO}_${N}_${TEMPERATURE}.json"
echo "  - output/${OUT_DIR_NAME}/${SCENARIO}_${N}_${TEMPERATURE}_eval.json"
echo "  - output/${OUT_DIR_NAME}/${SCENARIO}_${N}_${TEMPERATURE}_eval_all.json"
