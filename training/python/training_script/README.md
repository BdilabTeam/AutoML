# Huggingface model训练脚本
# Env Prepare:
```bash
# 激活虚拟环境
conda activate xxx / source ${VIRTUAL_ENV_PATH}/bin/activate
# 更新虚拟环境中的pip包
pip install --upgrade pip
# 在虚拟环境中安装poetry
pip install poetry
# 通过poetry进行依赖包安装
poetry install --extras storage
```
# Start huggingface_training_script
```
cd training/python/training_script/huggingface_training_script
```

```bash
python run_image_classification.py \
    --train_dir /Users/treasures/Downloads/image_classification_training/data/train \
    --model_name_or_path /Users/treasures/Downloads/image_classification_training/model \
    --output_dir /Users/treasures/Downloads/output \
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
```

```bash
# 增加pull_model_from_minio, training从minio中拉取模型进行训练
# 增加push_to_minio选项-训练完成后, 训练文件上传至minio server
python run_image_classification.py \
    --pull_model_from_minio True \
    --model_bucket_name automl \
    --model_object_name /pretrained-models/image_classification.zip \
    --model_storage_path /Users/treasures_y/Documents/test_minio/model/model.zip  \
    --train_dir /Users/treasures_y/Documents/image_classification_training/data/train \
    --output_dir /Users/treasures_y/Documents/test_minio/output \
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
    --push_to_minio True \
    --minio_endpoint 124.70.188.119:32090 \
    --access_key 42O7Ukrwo3lf9Cga3HZ9 \
    --secret_key ELN5mbp9kpzNPqeuM5iifpm8aLSqYlV57f7yVZqv \
    --archive_bucket_name automl \
    --archive_object_name /44/model.zip \
    --output_archive_dir /Users/treasures_y/Documents/test_minio/archive \
    --clean_archive_cache True
```