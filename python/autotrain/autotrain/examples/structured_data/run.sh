#!/usr/bin/env bash

set -x

# 获取当前脚本所在目录
DIR_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd $DIR_PATH

python run_densenet.py \
    --task_type "structured_data_classification" \
    --model_type "densenet" \
    --inputs "" \
    --dp_enable_auto_feature_extract True \
    --mp_num_layers "[1, 2, 3]" \
    --mp_num_units "[16, 32, 64, 128, 256, 512, 1024]" \
    --mp_use_batchnorm True \
    --mp_dropout "[0.0, 0.25, 0.5]" \
    --tp_directory "" \
    --tp_overwrite "True" \
    --tp_project_name "test" \
    --tp_max_trials 1 \
    --tp_objective "val_loss" \
    --tp_tuner "greedy" \
    --tp_batch_size 16 \
    --tp_epochs 1 \
    --tp_validation_split 0.3 \
    --tp_is_early_stop True \