from minio import Minio
from minio.error import MinioException
import shutil
from dataclasses import dataclass, field
from typing import Optional
import logging

minio_endpoint = '124.70.188.119:32090'  # MinIO 服务器的地址和端口
access_key = '42O7Ukrwo3lf9Cga3HZ9'
secret_key = 'ELN5mbp9kpzNPqeuM5iifpm8aLSqYlV57f7yVZqv'

logger = logging.getLogger(__name__)

# 创建一个压缩文件，将保存的模型文件添加到其中
def make_zip_archive(base_name: str, root_dir: str):
    shutil.make_archive(base_name=base_name, format='zip', root_dir=root_dir)

    
@dataclass
class MixinMinioClient(object):
    minio_endpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Minio Server 'Address:Port'"
        }
    )
    access_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Minio access key"
        }
    )
    secret_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Minio secret key"
        }
    )
    
    def __post_init__(self):
        if not self.minio_endpoint or not self.access_key or not self.secret_key:
            raise ValueError(
                f"If you want to create minio client, you must specify the following key words: minio_endpoint、access_key、secret_key\
                    currently, the endpoint is {self.minio_endpoint}, the access_key is {self.access_key}, the secret_key is {self.secret_key}"
            )
        # 创建 MinIO 客户端
        try:
            self._minio_client = Minio(self.minio_endpoint, self.access_key, self.secret_key, secure=False)
        except Exception as e:
            logger.exception(e)
            
    @property
    def minio_client(self) -> Minio:
        return self._minio_client