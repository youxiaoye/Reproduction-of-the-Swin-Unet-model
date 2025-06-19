#!/usr/bin/env bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 训练参数
python train.py \
    --output_dir ./model_out/ISIC2018 \
    --dataset ISIC2018 \
    --img_size 224 \
    --batch_size 32 \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path ./processed/ISIC2018 \
    --n_class 2 \
    --max_epochs 100 \
    --base_lr 0.01 \
    --num_workers 8 \
    --eval_interval 1
