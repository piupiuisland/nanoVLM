#!/bin/bash
#SBATCH --job-name=mathvista
#SBATCH --output=logs/eval_new/%A_%a.out
#SBATCH --error=logs/eval_new/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --array=0
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=luis.wiedmann@huggingface.co

# Change to project directory
cd /fsx/luis_wiedmann/nanoVLM
source .venv/bin/activate

# tasks=('ai2d') #ai2d chartqa docvqa infovqa mme mmmu mmstar ocrbench scienceqa textvqa seedbench

# runs=('/fsx/andi/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_36851samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-132458' \
# '/fsx/andi/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-091343')
# steps=(300 1500 2700 3900 5100 6300 7500 8700 9900 11100 12300 13500 14700 15900 17100 18300 19500 20700 21900 23100 24300 25500 26700 27900 29100 30300 31500 32700 33900 35100 36300 37500 38700 39900)
# python run_checkpoint_evaluations.py --checkpoints_dir ${runs[$SLURM_ARRAY_TASK_ID]} --eval_tasks ${tasks[@]} --steps ${steps[@]} --eval_results_dir eval_results_andi_new #--batch_size 32

# runs=('/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_7833samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-111329' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_14057samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-113306' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_3395samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-121358' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/fv_ss_unfiltered' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-165157' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-172025' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-173121' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-174041' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-205752' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-210619' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-105432' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-145130' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-130314' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-150042' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-165133' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-095710' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-100810' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-103222' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-131717' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-115740' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-075554' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-091630' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-083248' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-085529' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_46482samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-094301' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/base/step_10000/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-110408' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/fv_ss_unfiltered/step_20000/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-112516' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/fv_ss_unfiltered/step_20000/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-114701' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/fv_ss_unfiltered/step_20000/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-120558' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/fv_ss_unfiltered/step_20000/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-123023' \
# '/fsx/luis_wiedmann/nanoVLM/checkpoints/fv_ss_unfiltered/step_20000/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-132541')
# steps=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000)
# python run_checkpoint_evaluations.py --checkpoints_dir ${runs[$SLURM_ARRAY_TASK_ID]} --eval_tasks ${tasks[@]} --steps ${steps[@]} --eval_results_dir eval_results_new --batch_size 32


# MathVista works only in nanoVLM env (not in nanoVLM-2)
tasks=('seedbench')
runs=('/fsx/luis_wiedmann/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_60100_lr_vision_5e-05-language_5e-05-0.00512_0827-120356')
steps=(60000)
python run_checkpoint_evaluations.py --checkpoints_dir ${runs[@]} --eval_tasks ${tasks[@]} --steps ${steps[@]} --batch_size 4
