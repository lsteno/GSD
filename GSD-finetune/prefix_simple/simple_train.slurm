#!/bin/bash
#SBATCH --job-name=qwen1.5b-prefix
#SBATCH --output=logs/prefix_%j.out
#SBATCH --error=logs/prefix_%j.err
#SBATCH --partition=students
#SBATCH --nodelist=hpc-node08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

module load python/3.10.7

cd ~/GSD-finetune/prefix_simple
source ~/GSD-finetune/prefix_simple/.venv_simple/bin/activate

# force offline usage (assumes you already cached the model/dataset)
export HF_HOME="/home/s3221407/GSD-finetune/model_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Detect SLURM-allocated GPU
GPU_ID=$(echo $SLURM_JOB_GPUS | cut -d',' -f1)
if [ -z "$GPU_ID" ]; then
    GPU_ID=$(echo $SLURM_STEP_GPUS | cut -d',' -f1)
fi
if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

python3 train.py \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset data/opencodeinstruct_balanced_5k.json \
  --dev-dataset data/opencodeinstruct_balanced_200_dev.json \
  --output-dir runs/qwen2.5-1.5b-prefix-5k \
  --cache-dir /home/s3221407/GSD-finetune/prefix_simple/model_cache \
  --energy-log runs/qwen2.5-1.5b-prefix-5k/energy.json \
  --num-virtual-tokens 12 \
  --epochs 5 \
  --learning-rate 1e-3 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 2048 \
  --device-index $GPU_ID