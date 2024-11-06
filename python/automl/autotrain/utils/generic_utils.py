import os
import concurrent
from enum import Enum

class TaskType(Enum):
    STRUCTURED_DATA_CLASSIFICATION = "structured-data-classification"
    STRUCTURED_DATA_REGRESSION = "structured-data-regression"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_REGRESSION = "image-regression"

class ModelType(Enum):
    DENSENET = "densenet"
    RESNET = "resnet"

def upload_dir_to_minio(client, bucket_name, dir_path, prefix=''):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 构造 MinIO 中的路径
            minio_path = os.path.join(prefix, root.replace(dir_path, '')[1:], file)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future = executor.submit(
                    client.fput_object,
                    bucket_name=bucket_name,
                    object_name=minio_path,
                    file_path=os.path.join(root, file), 
                )
                future.add_done_callback(lambda _: print(f"上传成功: {minio_path}"))
            # try:
            #     # 上传文件到 MinIO
            #     client.fput_object(
            #         bucket_name=bucket_name,
            #         object_name=minio_path,
            #         file_path=os.path.join(root, file),
            #     )
            #     print(f"上传成功: {minio_path}")
            # except Exception as exc:
            #     print(f"上传失败: {minio_path}, 错误信息: {exc}")