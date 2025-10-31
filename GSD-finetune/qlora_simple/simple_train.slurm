#!/bin/bash
#SBATCH --job-name=qwen1.5b-qlora
#SBATCH --output=logs/qlora_%j.out
#SBATCH --error=logs/qlora_%j.err
#SBATCH --partition=students
#SBATCH --nodelist=hpc-node08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

module load python/3.10.7

cd ~/GSD-finetune/qlora_simple
source ~/GSD-finetune/qlora_simple/.venv_simple/bin/activate

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
    --dataset /home/s3221407/GSD-finetune/qlora_simple/data/opencodeinstruct_balanced_5k.json \
    --dev-dataset /home/s3221407/GSD-finetune/qlora_simple/data/opencodeinstruct_balanced_200_dev.json \
    --output-dir runs/1.5b-qlora-5k-q8 \
    --cache-dir /home/s3221407/GSD-finetune/model_cache \
    --energy-log runs/1.5b-qlora-5k-q8/energy.json \
    --epochs 5 \
    --learning-rate 4e-5 \
    --lora-r 8 \
    --lora-alpha 16 \
    --device-index $GPU_ID \
    --batch-size 2 \
    --grad-accum 16 \
    --max-length 2048 \
    --quant-bits 4 \

python3 merge.py \
    --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --lora-ckpt runs/1.5b-qlora-5k-q8 \
    --output-dir runs/1.5b-qlora-5k-q8-merged \
    --cache-dir /home/s3221407/GSD-finetune/model_cache
