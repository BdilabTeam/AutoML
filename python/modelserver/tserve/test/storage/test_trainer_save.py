# from transformers import Trainer, TrainingArguments
from minio import Minio

# # 设置 MinIO 服务器的连接信息
minio_endpoint = '124.70.188.119:32090'  # MinIO 服务器的地址和端口
access_key = '42O7Ukrwo3lf9Cga3HZ9'
secret_key = 'ELN5mbp9kpzNPqeuM5iifpm8aLSqYlV57f7yVZqv'

# # 创建 MinIO 客户端
minio_client = Minio(minio_endpoint, access_key, secret_key, secure=False)

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir='./output',
#     # 其他训练参数...
# )

# # 创建 Trainer
# trainer = Trainer(
#     model=model,  # 你的模型
#     args=training_args,
#     # 其他 Trainer 参数...
# )

# # 训练你的模型...

# # 保存模型到本地文件夹
# trainer.save_model(training_args.output_dir)

# # 上传模型文件到MinIO文件系统
# for file_name in os.listdir(training_args.output_dir):
#     file_path = os.path.join(training_args.output_dir, file_name)
#     minio_client.fput_object(bucket_name, f"{model_directory}/{file_name}", file_path)

# # 清理本地模型文件（可选）
# shutil.rmtree(training_args.output_dir)


# from transformers import Trainer, TrainingArguments

# # 创建 Trainer 和 TrainingArguments 实例
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     data_collator=data_collator,
# )

# # 进行模型训练
# trainer.train()

# # 保存训练得到的文件
# trainer.save_model("my_model")


import os
def list_dir():
    output_dir = "/Users/treasures/Downloads/pretrained_model/output/test"
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        minio_client.fput_object(bucket_name, f"{model_directory}/{file_name}", file_path)
# list_dir()

# # 创建一个压缩文件，将保存的模型文件添加到其中
import shutil
def make_archive():
    output_dir = "/Users/treasures/Downloads/output"
    project_id = "42"
    base_name = f"/Users/treasures/Downloads/{project_id}"
    shutil.make_archive(base_name, 'zip', root_dir=output_dir)
make_archive()

def push_to_minio():
    try:
        minio_client.fput_object("test", "/42/model.zip", "/Users/treasures/Downloads/output/42.zip", content_type="application/zip")
    except Exception as e:
        print(e)
# push_to_minio()