from urllib.parse import urlparse
import os

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
_is_offline_mode = True if os.environ.get("AUTOML_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False

def is_offline_mode():
    return _is_offline_mode

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
    for deprecated behavior allowing to download config/models with a single url instead of using the Hub.

    Args:
        url (`str`): The url of the file to download.

    Returns:
        `str`: The location of the temporary file where the url was downloaded.
    """
    # tmp_fd, tmp_file = tempfile.mkstemp()
    # with os.fdopen(tmp_fd, "wb") as f:
    #     http_get(url, f, proxies=proxies)
    # return tmp_file
    pass