#!/bin/bash
#SBATCH --job-name=eval_id
#SBATCH --output=logs/eval_id/%A_%a.out
#SBATCH --error=logs/eval_id/%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --gpus=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --array=0-1
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=luis.wiedmann@huggingface.co

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

tasks=('ai2d' 'chartqa') #mmstarmmmuocrbenchtextvqadocvqascienceqammeinfovqa #ai2d chartqa docvqa infovqa mme mmmu mmstar ocrbench scienceqa textvqa seedbench
runs=('/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_24103samples_bs256_80100_lr_vision_5e-05-language_5e-05-0.00512_0904-165925')
#steps=(1200 2400 3600 4800 6000 7200 8400 9600) 
#steps=(10800 12000 13200 14400 15600 16800 18000 19200)
steps=(80000)

# torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE run_checkpoint_evaluations.py --checkpoints_dir ${runs[$SLURM_ARRAY_TASK_ID]} --eval_tasks ${tasks[@]} --steps ${steps[@]} --eval_results_dir '/fsx/luis_wiedmann/nanoVLM/eval_results_andi'
python run_checkpoint_evaluations.py --checkpoints_dir ${runs[@]} --eval_tasks ${tasks[$SLURM_ARRAY_TASK_ID]} --steps ${steps[@]} --eval_results_dir '/fsx/luis_wiedmann/nanoVLM/eval_results'
