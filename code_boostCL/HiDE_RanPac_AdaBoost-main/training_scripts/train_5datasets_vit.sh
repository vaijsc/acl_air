#!/bin/bash

for seed in 42
do
CUDA_VISIBLE_DEVICES=1,2,7 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        --master_port='27500' \
        --use_env main.py \
        five_datasets_hideprompt_5e \
        --original_model vit_base_patch16_224 \
        --model vit_base_patch16_224 \
        --batch-size 32 \
        --data-path ./datasets \
        --epochs 20 \
        --sched constant \
        --seed $seed \
        --train_inference_task_only \
        --lr 0.001 \
        --output_dir "./output/5datasets_vit_multi_centroid_mlp_2_seed$seed" 
done



# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=1,2,7 python -m torch.distributed.launch \
#         --nproc_per_node=3 \
#         --use_env main.py \
#         five_datasets_hideprompt_5e \
#         --original_model vit_base_patch16_224 \
#         --model vit_base_patch16_224 \
#         --batch-size 32 \
#         --data-path ./datasets \
#         --output_dir ./output/5datasets_vit_pe_seed$seed \
#         --epochs 20 \
#         --sched constant \
#         --lr 0.03 \
#         --clip-grad 2 \
#         --reg 0.1 \
#         --prompt_momentum 0.01 \
#         --seed $seed \
#         --larger_prompt_lr \
#         --trained_original_model ./output/5datasets_vit_multi_centroid_mlp_2_seed$seed \
# done