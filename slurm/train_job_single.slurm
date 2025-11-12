#!/bin/bash
#SBATCH --job-name=train_nanoVLM
#SBATCH --output=logs/train_nanoVLM/%A_%a.out
#SBATCH --error=logs/train_nanoVLM/%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --gpus=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

torchrun --nproc_per_node=1 train.py