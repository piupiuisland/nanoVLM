#!/bin/bash\

#################
# Global Image Token or not
#################
python plot_eval_results.py \
    No-Token:'/fsx/luis_wiedmann/nanoVLM/eval_results_andi_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-131841' \
    Token:'/fsx/luis_wiedmann/nanoVLM/eval_results_andi_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-091343' \
    'Token&Resize':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-144934' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output global_image_token/global_image_token \
    --steps 300 1500 2700 3900 5100 6300 7500 8700 9900 11100 12300 13500 14700 15900 17100 18300 19500

#################
# Untie Head or not
#################
# TODO: Move this to eval_results_new
python plot_eval_results.py \
    'Tied LM Head':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_60100_lr_vision_5e-05-language_5e-05-0.00512_0827-120356' \
    'Untied LM Head':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0829-225348' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output untie/untie \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000

#################
# Experiments 4.1 (Against Baselines)
# TODO: Rerun Cauldron, Cambrian and LLaVa
#################
python plot_eval_results.py \
    FineVision:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    Cauldron:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_3395samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-121358' \
    Cambrian:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_14057samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-113306' \
    LLaVa:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_7833samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-111329' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output against_baselines/against_baselines \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

python plot_eval_results.py \
    'FineVision (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_45067samples_bs512_50100_lr_vision_5e-05-language_5e-05-0.00512_0828-163614' \
    'Cauldron (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_3489samples_bs512_40000_lr5e-05-0.00512_0811-092351' \
    'Cambrian (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_13814samples_bs512_40000_lr5e-05-0.00512_0811-101603' \
    'LLaVa (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_7726samples_bs512_40000_lr5e-05-0.00512_0811-130750' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output against_baselines/against_baselines_deduplicated \
    --steps 1200 2400 3600 4800 6000 7200 8400 9600 10800 12000 13200 14400 15600 16800 18000 19200

python plot_eval_results.py \
    Cauldron:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_3395samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-121358' \
    'Cauldron (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_3489samples_bs512_40000_lr5e-05-0.00512_0811-092351' \
    --tasks 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average' \
    --output against_baselines/cauldron_dedup \
    --steps 300 2700 5100 7500 9900 11400 14700 17100 19500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

python plot_eval_results.py \
    Cambrian:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_14057samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-113306' \
    'Cambrian (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_13814samples_bs512_40000_lr5e-05-0.00512_0811-101603' \
    --tasks 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average' \
    --output against_baselines/cambrian_dedup \
    --steps 300 2700 5100 7500 9900 11400 14700 17100 19500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

python plot_eval_results.py \
    LLaVa:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_7833samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0823-111329' \
    'LLaVa (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_7726samples_bs512_40000_lr5e-05-0.00512_0811-130750' \
    --tasks 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average' \
    --output against_baselines/llava_dedup \
    --steps 300 2700 5100 7500 9900 11400 14700 17100 19500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

python plot_eval_results.py \
    FineVision:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    'FineVision (DD)':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_45067samples_bs512_50100_lr_vision_5e-05-language_5e-05-0.00512_0828-163614' \
    --tasks 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average' \
    --output against_baselines/finevision_dedup \
    --steps 300 2700 5100 7500 9900 11400 14700 17100 19500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 1200 2400 3600 4800 6000 7200 8400 9600 10800 12000 13200 14400 15600 16800 18000 19200

#################
# Experiments 4.b (Internal Deduplication)
# TODO: Run additional Benchmarks
#################
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_andi_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-091343' \
    'Internal Deduplication':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_36851samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-132458' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output internal_deduplication/internal_deduplication \
    --steps 300 2700 5100 7500 9900 12300 14700 17100 19500 21900 24300 26700 29100 31500 33900 36300 38700 #1500 3900 6300 8700 11100 13500 15900 18300 20700  23100 25500 27900 30300 32700 35100 37500 39900

#################
# Experiments 4.c (Remove other languages)
#################
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    'Remove Multilingual Data':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_46482samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-094301' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output remove_ch/remove_ch \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000

#################
# Experiments 4.d) i) (Individual ratings)
#################

# Plot Relevance Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    '≥2':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-165157' \
    '≥3':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-172025' \
    '≥4':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-173121' \
    '≥5':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-174041' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output fl_relevance/relevance_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

# Plot Image Correspondence Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    '≥2':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-205752' \
    '≥3':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-210619' \
    '≥4':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-105432' \
    '≥5':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-145130' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output fl_image_correspondence/image_correspondence_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000

# Plot Visual Dependency Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    '≥2':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-130314' \
    '≥3':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-150042' \
    '≥4':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-165133' \
    '≥5':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-095710' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output fl_visual_dependency/visual_dependency_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000

# Plot Formatting Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    '≥2':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-100810' \
    '≥3':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-103222' \
    '≥4':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-131717' \
    '≥5':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-115740' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output fl_formatting/formatting_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

#################
# Experiments 4.d) ii) (All ratings)
# TODO: Rerun with proper setup
#################
# Andi's runs
# python plot_eval_results.py \
#     'All_Samples':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0812-110026' \
#     '>=2':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-063155' \
#     '>=3':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0812-004500' \
#     '>=4':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0812-033512' \
#     '>=5':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0815-082051' \
#     --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average' 'average_rank' \
#     --output all_ratings/all_ratings_andi
    
    #'>=4cont':'/fsx/andi/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-081736' \

# My runs
python plot_eval_results.py \
    'Baseline':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    '≥2':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-075554' \
    '≥3':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-091630' \
    '≥4':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-083248' \
    '≥5':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0822-085529' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 \
    --output all_ratings/all_ratings_luis

#################
# Experiments 4.e) (Multiple Stages)
# TODO: Rerun with proper setup
#################
# Andi's runs
# python plot_eval_results.py \
#     '1_on_top_1':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-081736' \
#     '3_on_top_1':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-101418' \
#     '5_on_top_1':'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-125149' \
#     --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average' 'average_rank' \
#     --output multi_stage/multi_stage_andi

# Stage-1 vs not
python plot_eval_results.py \
    'Single Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/fv_ss_unfiltered' \
    'Two Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-110408' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 \
    --output multi_stage/ss_vs_s1

# TODO: Move to eval_results_new
python plot_eval_results.py \
    'Single Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_60100_lr_vision_5e-05-language_5e-05-0.00512_0827-120356' \
    'Two Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_60100_lr_vision_5e-05-language_5e-05-0.00512_0901-105355' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --steps 1000 3000 5000 7000 9000 11000 13000 15000 17000 19000 21000 23000 25000 27000 29000 31000 33000 35000 37000 39000 41000 43000 45000 47000 49000 51000 53000 55000 57000 59000 \
    --output multi_stage/ss_vs_s1_fullres
    
    #--steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000 41000 42000 43000 44000 45000 46000 47000 48000 49000 50000 51000 52000 53000 54000 55000 56000 57000 58000 59000 60000 \

python plot_eval_results.py \
    'Single Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-135M_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0828-173721' \
    'Two Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-135M_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0829-094924' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 \
    --output multi_stage/ss_vs_s1_230M

python plot_eval_results.py \
    'Single Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-so400m-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0829-124251' \
    'Two Stage':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-so400m-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0829-135307' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 \
    --output multi_stage/ss_vs_s1_800M

# Stage2.5 with Ratings
python plot_eval_results.py \
    '≥1':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-112516' \
    '≥2':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-114701' \
    '≥3':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-120558' \
    '≥4':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-123023' \
    '≥5':'/fsx/luis_wiedmann/nanoVLM/eval_results_new/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0824-132541' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 \
    --output multi_stage/s25_ratings

#################
# Experiments 5) (Model Max)
#################

python plot_eval_results.py \
    '460M':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_60100_lr_vision_5e-05-language_5e-05-0.00512_0827-120356' \
    --tasks 'seedbench_seed_all' 'chartqa_relaxed_overall' 'docvqa_val_anls' 'infovqa_val_anls' 'mme_total_score' 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'scienceqa_exact_match' 'textvqa_val_exact_match' 'ai2d_exact_match' 'average_rank' 'average' \
    --output modelmax/450M
