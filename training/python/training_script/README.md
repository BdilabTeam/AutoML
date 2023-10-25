# Huggingface model训练脚本
# Start image_classification_training_script
```
cd training/python/training_script/image_classification_training_script
```

```bash
python run_image_classification.py \
    --train_dir /Users/treasures/Downloads/image_classification_training/data/train \
    --model_name_or_path /Users/treasures/Downloads/image_classification_training/model \
    --output_dir /Users/treasures/Downloads/image_classification_training/output \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --ignore_mismatched_sizes True
```