#!/bin/bash
#SBATCH --job-name=train_nanoVLM
#SBATCH --output=logs/train_nanoVLM/%A_%a.out
#SBATCH --error=logs/train_nanoVLM/%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --gpus=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --array=0-17

cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

# --- Define configuration arrays ---
declare -a lr_mps=("0.01" "0.008" "0.005" "0.003" "0.001" "0.01" "0.008" "0.005" "0.003" "0.001" "0.01" "0.008" "0.005" "0.003" "0.001" "0.00512" "0.00512" "0.00512")
declare -a lr_backbones=("1e-4" "1e-4" "1e-4" "1e-4" "1e-4" "5e-5" "5e-5" "5e-5" "5e-5" "5e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "1e-4" "2e-5" "1e-5")

# --- Get parameters for the current job task ---
current_lr_mp=${lr_mps[$SLURM_ARRAY_TASK_ID]}
current_lr_backbone=${lr_backbones[$SLURM_ARRAY_TASK_ID]}

# --- Run the training script with parameters ---
python train.py \
    --lr_mp $current_lr_mp \
    --lr_backbones $current_lr_backbone \
    --compile True \