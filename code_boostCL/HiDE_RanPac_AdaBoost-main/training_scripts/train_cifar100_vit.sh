#!/bin/bash

# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --master_port='26507'\
#         --use_env main.py \
#         cifar100_hideprompt_5e \
#         --original_model vit_base_patch16_224 \
#         --model vit_base_patch16_224 \
#         --batch-size 24 \
#         --data-path ../Z.Data/ \
#         --output_dir ./output/cifar100_sup21k_multi_centroid_mlp_2_seed$seed \
#         --epochs 20 \
#         --sched constant \
#         --seed $seed \
#         --train_inference_task_only \
#         --lr 0.0005 
# done

# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
# 	--nproc_per_node=1 \
# 	--master_port='28605' \
# 	--use_env main.py \
# 	cifar100_hideprompt_5e \
# 	--model vit_base_patch16_224 \
# 	--original_model vit_base_patch16_224 \
# 	--batch-size 24 \
# 	--epochs 50 \
# 	--data-path ../Z.Data/ \
# 	--ca_lr 0.005 \
# 	--crct_epochs 30 \
# 	--seed $seed \
# 	--prompt_momentum 0.01 \
# 	--reg 0.1 \
# 	--length 5 \
# 	--sched step \
# 	--larger_prompt_lr \
# 	--trained_original_model ./output/cifar100_sup21k_multi_centroid_mlp_2_seed$seed \
# 	--output_dir ./output/cifar100_vit_pe_seed$seed\
# done

for seed in 42
do
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='28605' \
	--use_env main.py \
	cifar100_hideprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 256 \
	--epochs 50 \
	--data-path ../Z.Data/ \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed $seed \
	--prompt_momentum 0.01 \
	--reg 0.1 \
	--length 5 \
	--sched step \
	--larger_prompt_lr \
	--trained_original_model ./output/cifar100_sup21k_multi_centroid_mlp_2_seed$seed \
	--output_dir ./output/cifar100_vit_pe_seed$seed\
	--wols
done
