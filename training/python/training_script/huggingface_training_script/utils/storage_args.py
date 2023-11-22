from dataclasses import dataclass, field
from typing import Optional
import shutil
import os

MODEL_ARCHIVE_NAME = "model.zip"

@dataclass
class StorageArguments(object):
    # minio配置
    minio_endpoint: str = field(
        default=None,
        metadata={
            "help": "Minio Server 'Address:Port'"
        }
    )
    access_key: str = field(
        default=None,
        metadata={
            "help": "Minio access key"
        }
    )
    secret_key: str = field(
        default=None,
        metadata={
            "help": "Minio secret key"
        }
    )

    # training将output目录下的文件保存为zip存档并推送至minio依赖配置
    push_to_minio: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable minio storage. If enabled, you must specify the following key words: endpoint、access_key、secret_key"
        }
    )
    archive_bucket_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the Minio bucket to upload or download"
        }
    )
    archive_object_name: str = field(
        default= None,
        metadata={
            "help": f"Name of the object stored in the {archive_bucket_name} bucke. This field will be generated automatically"
        }
    )
    # project_id: int = field(
    #     default=None,
    #     metadata={
    #         "help": "The ID of the training project"
    #     }
    # )
    output_archive_dir: str = field(
        default="/training_script/huggingface_training_script/output_archive",
        metadata={
            "help": "Storage Archive Dir. output -> .zip -> archive_dir"
        }
    )
    clean_archive_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to delete archive cache?"
        }
    )
    # 从minio拉取模型文件
    pull_model_from_minio: bool = field(
        default=False,
        metadata={
            "help": "Pull the model file from the minio file system to directory xxx"
        }
    )
    model_bucket_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the bucket where the model is stored"
        }
    )
    model_object_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the object in the model bucket"
        }
    )
    model_storage_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory for storing model files pulled from the minio file system. eg. /training_script/huggingface_training_script/model/model.zip"
        }
    )
    # 从minio拉取数据文件
    pull_data_from_minio: bool = field(
        default=False,
        metadata={
            "help": "Pull the data file from the minio file system to directory xxx"
        }
    )
    data_bucket_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the bucket where the data is stored"
        }
    )
    data_object_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the object in the data bucket"
        }
    )
    data_storage_path: Optional[str] = field(
        # default="/training_script/huggingface_training_script/data.zip",
        default=None,
        metadata={
            "help": "Directory for storing data files pulled from the minio file system.  eg. /training_script/huggingface_training_script/data/data.zip"
        }
    )
    
    def __post_init__(self):
        if self.push_to_minio or self.pull_model_from_minio or self.pull_data_from_minio:
            if not self.minio_endpoint or not self.access_key or not self.secret_key:
                raise ValueError(
                    f"If you enabled minio storage, you must specify the following key words: endpoint、access_key、secret_key\
                        currently, the endpoint is {self.minio_endpoint}, the access_key is {self.access_key}, the secret_key is {self.secret_key}"
                )
            # 推送至minio文件系统所依赖的参数
            if self.push_to_minio:
                if not self.archive_bucket_name:
                    raise ValueError(
                        f"If you want to push the model file to minio, you must specify the key words: archive_bucket_name\
                            currently, the archive_bucket_name is {self.archive_bucket_name}"
                    )
                if not self.archive_object_name:
                    raise ValueError(
                        f"If you want to push the model file to minio, you must specify the key words: archive_object_name\
                            currently, the archive_object_name is {self.archive_object_name}"
                    )
                # if not self.project_id:
                #     raise ValueError(
                #         f"If you enabled minio storage, you must specify the key words: project_id\
                #             currently, the project_id is {self.project_id}"
                #     )
                # Check output_archive_dir validity
                if os.path.exists(self.output_archive_dir):
                    shutil.rmtree(self.output_archive_dir)
                else:
                    os.makedirs(self.output_archive_dir)
                # 用于shutil.make_archive()，此方法base_name参数要求'无'扩展名
                self.output_archive_path_without_zip_extension = f"{self.output_archive_dir}/model"
                self.output_archive_path = f"{self.output_archive_dir}/{MODEL_ARCHIVE_NAME}"
                
                # self.archive_object_name = f"{self.project_id}/{MODEL_ARCHIVE_NAME}"
            if self.pull_model_from_minio:
                if not self.model_bucket_name:
                    raise ValueError(
                        f"If you want to pull the model file from minio, you must specify the key words: model_bucket_name\
                            currently, the model_bucket_name is {self.model_bucket_name}"
                    )
                if not self.model_object_name:
                    raise ValueError(
                        f"If you want to pull the model file from minio, you must specify the key words: model_object_name\
                            currently, the model_object_name is {self.model_object_name}"
                    )
                if not self.model_storage_path:
                    raise ValueError(
                        f"If you enabled pulling model file from minio storage, you must specify the key words: model_storage_path\
                            currently, the model_storage_path is {self.model_storage_path}"
                    )
                # Check model_storage_path validity
                model_storage_dir = os.path.dirname(self.model_storage_path)
                if os.path.exists(model_storage_dir):
                    shutil.rmtree(model_storage_dir)
                # if os.path.exists(self.model_storage_path):
                #     if os.path.isfile(self.model_storage_path):
                #         os.remove(self.model_storage_path)
                #     elif os.path.isdir(self.model_storage_path):
                #         os.rmdir(self.model_storage_path)
                # else:
                #     os.makedirs(self.model_storage_path)
            if self.pull_data_from_minio:
                if not self.data_bucket_name:
                    raise ValueError(
                        f"If you want to pull the data file from minio, you must specify the key words: data_bucket_name\
                            currently, the data_bucket_name is {self.data_bucket_name}"
                    )
                if not self.data_object_name:
                    raise ValueError(
                        f"If you want to pull the data file from minio, you must specify the key words: data_object_name\
                            currently, the data_object_name is {self.data_object_name}"
                    )
                if not self.data_storage_path:
                    raise ValueError(
                        f"If you enabled pulling data file from minio storage, you must specify the key words: data_storage_path\
                            currently, the data_storage_path is {self.data_storage_path}"
                    )
                # Check data_storage_path validity
                model_storage_dir = os.path.dirname(self.data_storage_path)
                if os.path.exists(self.data_storage_path):
                    shutil.rmtree(self.data_storage_path)
                # if os.path.exists(self.data_storage_path):
                #     if os.path.isfile(self.data_storage_path):
                #         os.remove(self.data_storage_path)
                #     elif os.path.isdir(self.data_storage_path):
                #         os.rmdir(self.data_storage_path)
                # else:
                #     os.makedirs(self.data_storage_path)
                