import os
import pytest
from minio import Minio
from minio.deleteobjects import DeleteObject
import concurrent

MINIO_ENDPOINT = "124.70.188.119:32090"
MINIO_ACCESS_KEY= "lUAFGmD6TL57zCcFJTmo"
MINIO_SECRET_KEY = "ZMIG5DToDNtL5I86oeEDcvPhIE5PhlFe67oMVN0a"

BUCKET_NAME = "automl"
OBEJCT_NAME = "test"
FOLDER_PATH = "/Users/treasures_y/Documents/code/HG/AutoML/python/automl/autotrain/datasets/image-classification"
class TestMinio:
    @pytest.fixture
    def minio_client(self):
        return Minio(
            endpoint=MINIO_ENDPOINT, 
            access_key=MINIO_ACCESS_KEY, 
            secret_key=MINIO_SECRET_KEY, 
            secure=False
        )
        
    def test_push_dir(self, minio_client: Minio):
        def upload_dir_to_minio(client: Minio, bucket_name, dir_path, prefix=''):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    # 构造 MinIO 中的路径
                    minio_path = os.path.join(prefix, root.replace(dir_path, '')[1:], file)
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
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        future = executor.submit(
                            client.fput_object,
                            bucket_name=bucket_name,
                            object_name=minio_path,
                            file_path=os.path.join(root, file)
                        )
                        future.add_done_callback(lambda _: print(f"上传成功: {minio_path}"))
        
        upload_dir_to_minio(client=minio_client, bucket_name=BUCKET_NAME, dir_path=FOLDER_PATH, prefix="test/datasets")
    
    def test_del_dir(self, minio_client: Minio):
        dir_path = FOLDER_PATH
        prefix="test/datasets"
        object_list = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # 构造 MinIO 中的路径
                minio_path = os.path.join(prefix, root.replace(dir_path, '')[1:], file)
                object_list.append(minio_path)
    
        minio_client.remove_objects(bucket_name=BUCKET_NAME, delete_object_list=object_list)
        
        delete_object_list = list(
            map(
                lambda x: DeleteObject(x.object_name),
                minio_client.list_objects(
                    BUCKET_NAME,
                    "test",
                    recursive=True,
                ),
            )
        )
        errors = minio_client.remove_objects(BUCKET_NAME, delete_object_list)
        for error in errors:
            print("error occurred when deleting object", error)