#!/usr/bin/env bash

set -x

# 获取当前脚本所在目录
DIR_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd $DIR_PATH

python run_resnet.py \
    --task_type "image_classification" \
    --model_type "resnet" \
    --inputs "" \
    --tp_directory "" \
    --tp_overwrite "True" \
    --tp_project_name "test" \
    --tp_max_trials 1 \
    --tp_objective "val_loss" \
    --tp_tuner "greedy" \
    --tp_batch_size 32 \
    --tp_epochs 1 \
    --tp_validation_split 0.3 \
    --tp_is_early_stop True \

 