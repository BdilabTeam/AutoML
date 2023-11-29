#!/usr/bin/env bash

set -x

# 获取当前脚本所在目录
DIR_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd $DIR_PATH

python run_image_classification.py \
    --train_dir /Users/treasures_y/Documents/image_classification_training/data/train \
    --model_name_or_path /Users/treasures_y/Documents/image_classification_training/model \
    --output_dir /Users/treasures_y/Documents/image_classification_training/output \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --seed 1337 \
    --ignore_mismatched_sizes True \
    --use_cpu True \