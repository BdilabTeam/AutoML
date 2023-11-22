from huggingface_training_script.utils.storage_args import StorageArguments
import shutil
import logging
from huggingface_training_script.utils.minio_client_utils import MixinMinioClient, MinioException
from pathlib import Path

logger = logging.getLogger(__name__)
project_id = 42
storage_args = StorageArguments(
    minio_endpoint="124.70.188.119:32090",
    access_key="42O7Ukrwo3lf9Cga3HZ9",
    secret_key="ELN5mbp9kpzNPqeuM5iifpm8aLSqYlV57f7yVZqv",
    pull_model_from_minio=True,
    data_bucket_name="automl",
    data_object_name=f"/{project_id}/data.zip",
    data_storage_path="/Users/treasures_y/Documents/test_minio/data.zip",
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
        logger.info(f"Start to pull data archive from minio server")
        minio_clent.fget_object(
            bucket_name=storage_args.data_bucket_name,
            object_name=storage_args.data_object_name,
            file_path=storage_args.data_storage_path,
        )
        logger.info("Success to pull data archive from minio server")
    except MinioException as e:
        logger.exception("Failed to pull data archive from minio server")