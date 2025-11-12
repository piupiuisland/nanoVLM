#!/bin/bash
#SBATCH --job-name=train_nanoVLM_torchrun
#SBATCH --output=logs/train_nanoVLM/%A.out
#SBATCH --error=logs/train_nanoVLM/%A.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM

# Activate virtual environment
source .venv/bin/activate

# Run using torchrun
torchrun --nproc_per_node=8 train.py