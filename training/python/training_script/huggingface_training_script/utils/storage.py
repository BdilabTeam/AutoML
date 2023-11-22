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