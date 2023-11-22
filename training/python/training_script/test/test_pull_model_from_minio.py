from huggingface_training_script.utils.storage_args import StorageArguments
from huggingface_training_script.utils.storage import Storage
from huggingface_training_script.utils.minio_client_utils import MixinMinioClient, MinioException
import os
import logging


logger = logging.getLogger(__name__)

def test_pull_model_from_minio():
    task_type = "image_classification"
    storage_args = StorageArguments(
        minio_endpoint="124.70.188.119:32090",
        access_key="42O7Ukrwo3lf9Cga3HZ9",
        secret_key="ELN5mbp9kpzNPqeuM5iifpm8aLSqYlV57f7yVZqv",
        pull_model_from_minio=True,
        model_bucket_name="automl",
        model_object_name=f"/pretrained-models/{task_type}.zip",
        model_storage_path=f"/Users/treasures_y/Documents/test_minio/model/{task_type}.zip",
    )

    if storage_args.pull_model_from_minio:

        # 创建minio client
        minio_clent = MixinMinioClient(
            minio_endpoint=storage_args.minio_endpoint,
            access_key=storage_args.access_key,
            secret_key=storage_args.secret_key
        ).minio_client
        # 使用minio client将存档上传至minio server
        try:
            logger.info(f"Start to pull model archive from minio server")
            minio_clent.fget_object(
                bucket_name=storage_args.model_bucket_name,
                object_name=storage_args.model_object_name,
                file_path=storage_args.model_storage_path,
            )
            logger.info("Success to pull model archive from minio server")
        except MinioException as e:
            logger.exception("Failed to pull model archive from minio server")
        # 解压缩zip文件
        Storage._unpack_archive_file(
            file_path=storage_args.model_storage_path,
            mimetype="application/zip",
        )

def test_unpack_archive_file():
    file_path = "/Users/treasures_y/Documents/test_minio/model/image_classification.zip"
    target_dir = os.path.dirname(file_path)
    # 解压缩zip文件
    Storage._unpack_archive_file(
        file_path=file_path,
        mimetype="application/zip",
        target_dir=target_dir
    )

if __name__=="__main__":
    test_pull_model_from_minio()
    # test_unpack_archive_file()
    pass