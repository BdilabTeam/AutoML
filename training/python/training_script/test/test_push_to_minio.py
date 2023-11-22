from huggingface_training_script.utils.storage_args import StorageArguments
import shutil
import logging
from huggingface_training_script.utils.minio_client_utils import MixinMinioClient, MinioException
from pathlib import Path

logger = logging.getLogger(__name__)
projcet_id = 43
storage_args = StorageArguments(
    minio_endpoint="124.70.188.119:32090",
    access_key="42O7Ukrwo3lf9Cga3HZ9",
    secret_key="ELN5mbp9kpzNPqeuM5iifpm8aLSqYlV57f7yVZqv",
    push_to_minio=True,
    archive_bucket_name="automl",
    archive_object_name=f"/{projcet_id}/model.zip",
    output_archive_dir="/Users/treasures_y/Documents/test_minio/archive",
    # clean_archive_cache=True
)


if storage_args.push_to_minio:

    # 将训练输出的文件存档为zip
    shutil.make_archive(
        base_name=storage_args.output_archive_path_without_zip_extension, 
        format="zip", 
        root_dir="/Users/treasures_y/Documents/test_minio/output"
    )
    # 创建minio client
    minio_clent = MixinMinioClient(
        minio_endpoint=storage_args.minio_endpoint,
        access_key=storage_args.access_key,
        secret_key=storage_args.secret_key
    ).minio_client
    # 使用minio client将存档上传至minio server
    try:
        logger.info(f"Start to upload archive to minio server")
        minio_clent.fput_object(
            bucket_name=storage_args.archive_bucket_name,
            object_name=storage_args.archive_object_name,
            file_path=storage_args.output_archive_path,
            content_type="application/zip"
        )
        logger.info("Success to upload archive to minio server")
    except MinioException as e:
        logger.exception("Failed to push to minio server")
    
    if storage_args.clean_archive_cache:
        # 清理本地模型文件
        shutil.rmtree(storage_args.archive_path)