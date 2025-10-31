#!/bin/bash
#SBATCH --job-name=qwen1.5b-bitfit
#SBATCH --output=logs/bitfit_%j.out
#SBATCH --error=logs/bitfit_%j.err
#SBATCH --partition=students
#SBATCH --nodelist=hpc-node08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

module load python/3.10.7

cd ~/GSD-finetune/bitfit_simple
source ~/GSD-finetune/bitfit_simple/.venv_simple/bin/activate

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
    --output-dir runs/1.5b-bitfit-5k \
    --cache-dir /home/s3221407/GSD-finetune/bitfit_simple/model_cache \
    --energy-log runs/1.5b-bitfit-5k/energy.json \
    --epochs 5 \
    --learning-rate 4e-4 \
    --batch-size 2 \
    --grad-accum 16 \
    --max-length 2048 \
    --device-index $GPU_ID

python3 merge.py \
    --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --bitfit-ckpt runs/1.5b-bitfit-5k \
    --output-dir runs/1.5b-bitfit-5k-merged \
    --cache-dir /home/s3221407/GSD-finetune/model_cache
