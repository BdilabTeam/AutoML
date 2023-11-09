from dataclasses import dataclass, field
from typing import Optional

MODEL_ARCHIVE_NAME = "model.zip"

@dataclass
class StorageArguments(object):
    push_to_minio: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable minio storage. If enabled, you must specify the following key words: endpoint、access_key、secret_key"
        }
    )
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
    project_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "The ID of the training project"
        }
    )
    bucket_name: Optional[str] = field(
        default="automl",
        metadata={
            "help": "Name of the MinIO bucket to be uploaded"
        }
    )
    object_name: Optional[str] = field(
        default= None,
        metadata={
            "help": f"Name of the object stored in the {bucket_name} bucket, "
        }
    )
    archive_path: Optional[str] = field(
        default="/archive",
        metadata={
            "help": "Storage Archive Path"
        }
    )
    clean_archive_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to delete archive cache?"
        }
    )
    
    def __post_init__(self):
        if self.push_to_minio:
            if not self.minio_endpoint or not self.access_key or not self.secret_key:
                raise ValueError(
                    f"If you enabled minio storage, you must specify the following key words: endpoint、access_key、secret_key\
                        currently, the endpoint is {self.minio_endpoint}, the access_key is {self.access_key}, the secret_key is {self.secret_key}"
                )
            if not self.project_id:
                raise ValueError(
                    f"If you enabled minio storage, you must specify the key words: project_id\
                        currently, the project_id is {self.project_id}"
                )
            if not self.archive_path:
                raise ValueError(
                    f"If you enabled minio storage, you must specify the key words: archive_path\
                        currently, the archive_path is {self.archive_path}"
                )
            self.archive_file_path_without_zip_extension = f"{self.archive_path}/{self.project_id}"
            self.archive_file_path = f"{self.archive_path}/{self.project_id}.zip"
            self.object_name = f"{self.project_id}/{MODEL_ARCHIVE_NAME}"