import os
import re
import logging
import json
import gzip
import tarfile
import zipfile
import requests
import shutil
import mimetypes
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_URI_RE = "https?://(.+)/(.+)"
_HTTP_PREFIX = "http(s)://"
_HEADERS_SUFFIX = "-headers"

class Storage(object):  # pylint: disable=too-few-public-methods
    @staticmethod
    def download(uri: str, out_dir: str = None) -> str:
        if re.search(_URI_RE, uri):
            return Storage._download_from_uri(uri, out_dir)
        else:
            raise Exception("Cannot recognize storage type for " + uri +
                            "\n'%s' are the current available storage type." %
                            (_HTTP_PREFIX))

        logger.info("Successfully copied %s to %s", uri, out_dir)
        return out_dir
    
    @staticmethod
    def _download_from_uri(uri, out_dir=None):
        url = urlparse(uri)
        filename = os.path.basename(url.path)
        # Determine if the symbol '?' exists in the path
        if mimetypes.guess_type(url.path)[0] is None and url.query != '':
            mimetype, encoding = mimetypes.guess_type(url.query)
        else:
            mimetype, encoding = mimetypes.guess_type(url.path)
        local_path = os.path.join(out_dir, filename)

        if filename == '':
            raise ValueError('No filename contained in URI: %s' % (uri))

        # Get header information from host url
        headers = {}
        host_uri = url.hostname

        headers_json = os.getenv(host_uri + _HEADERS_SUFFIX, "{}")
        headers = json.loads(headers_json)

        with requests.get(uri, stream=True, headers=headers) as response:
            if response.status_code != 200:
                raise RuntimeError("URI: %s returned a %s response code." % (uri, response.status_code))
            zip_content_types = ('application/x-zip-compressed', 'application/zip', 'application/zip-compressed')
            if mimetype == 'application/zip' and not response.headers.get('Content-Type', '') \
                    .startswith(zip_content_types):
                raise RuntimeError("URI: %s did not respond with any of following \'Content-Type\': " % uri +
                                   ", ".join(zip_content_types))
            tar_content_types = ('application/x-tar', 'application/x-gtar', 'application/x-gzip', 'application/gzip')
            if mimetype == 'application/x-tar' and not response.headers.get('Content-Type', '') \
                    .startswith(tar_content_types):
                raise RuntimeError("URI: %s did not respond with any of following \'Content-Type\': " % uri +
                                   ", ".join(tar_content_types))
            if (mimetype != 'application/zip' and mimetype != 'application/x-tar') and \
                    not response.headers.get('Content-Type', '').startswith('application/octet-stream'):
                raise RuntimeError("URI: %s did not respond with \'Content-Type\': \'application/octet-stream\'"
                                   % uri)

            if encoding == 'gzip':
                stream = gzip.GzipFile(fileobj=response.raw)
                local_path = os.path.join(out_dir, f'{filename}.tar')
            else:
                stream = response.raw
            with open(local_path, 'wb') as out:
                shutil.copyfileobj(stream, out)

        if mimetype in ["application/x-tar", "application/zip"]:
            Storage._unpack_archive_file(local_path, mimetype, out_dir)

        return out_dir
    
    @staticmethod
    def _unpack_archive_file(file_path, mimetype, target_dir=None):
        if not target_dir:
            target_dir = os.path.dirname(file_path)

        try:
            logging.info("Unpacking: %s", file_path)
            if mimetype == "application/x-tar":
                archive = tarfile.open(file_path, 'r', encoding='utf-8')
            else:
                archive = zipfile.ZipFile(file_path, 'r')
            archive.extractall(target_dir)
            archive.close()
        except (tarfile.TarError, zipfile.BadZipfile):
            raise RuntimeError("Failed to unpack archive file. \
The file format is not valid.")
        os.remove(file_path)
    
    @staticmethod
    def _pull_from_minio(minio_client, bucket_name: str, object_name: str, file_path: str):
        try:
            minio_client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
            )
        except Exception as e:
            logger.exception("Failed to pull model archive from minio")
            raise Exception(e)
    
    @staticmethod
    def _push_to_minio(minio_client, bucket_name: str, object_name: str, file_path: str, content_type: str):
        try:
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type
            )
        except Exception as e:
            logger.exception("Failed to push to minio server")
            raise Exception(e)


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
                if os.path.exists(self.model_storage_dir):
                    shutil.rmtree(self.model_storage_dir)
                # if os.path.exists(self.data_storage_path):
                #     if os.path.isfile(self.data_storage_path):
                #         os.remove(self.data_storage_path)
                #     elif os.path.isdir(self.data_storage_path):
                #         os.rmdir(self.data_storage_path)