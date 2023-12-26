#!/usr/bin/env bash

set -x

# 获取当前脚本所在目录
DIR_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd $DIR_PATH

python run_densenet.py \
    --task_type "structured_data_classification" \
    --model_type "densenet" \
    --train_dir "/root/workspace/YJX/auto-ml/automl/python/autotrain/autotrain/datasets/train.csv" \
    --output_dir "/root/workspace/YJX/auto-ml/automl/python/autotrain/tests/output" \
    --overwrite "True" \
    --project_name "test" \
    --max_trials 2 \
    --objective "val_loss" \
    --tuner "greedy" \
    --batch_size 32 \
    --epochs 5 \
    --validation_split 0.3 \
    --is_early_stop True \
    --do_auto_feature_extract True \
    --do_auto_hyperparameter_tuning True \
    --num_layers_search_space "[1, 2, 3]" \
    --num_units_search_space "[16, 32, 64, 128, 256, 512, 1024]" \
    --use_batchnorm True \
    --dropout_space_search_space "[0.0, 0.25, 0.5]" \
    --iters 1
    