import sys
import os
import importlib
import inspect

from typing import Type, no_type_check
from pydantic import Field, BaseSettings as _BaseSettings
from contextlib import contextmanager

from .version import __version__

ENV_FILE_SETTINGS = ".env"
ENV_PREFIX_SETTINGS = "AUTOML_SERVER_"

DEFAULT_PARALLEL_WORKERS = 1

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")
PROMPT_TEMPLATE_FILE_PATH = os.path.join(PARENT_DIR, 'autoselect', 'resources', 'prompt-templates', 'model-selection-prompt-v1.json')
MODEL_METADATA_FILE_PATH = os.path.join(PARENT_DIR, 'autoselect', 'resources', 'automl-models-metadata.jsonl')

HOST_INFO_FILE_PATH = os.path.join(PARENT_DIR, 'autoschedule', 'host_info.json')
KUBE_CONFIG_FILE_PATh = "/root/.kube/config"

@contextmanager
def _extra_sys_path(extra_path: str):
    sys.path.insert(0, extra_path)

    yield

    sys.path.remove(extra_path)


def _get_import_path(klass: Type):
    return f"{klass.__module__}.{klass.__name__}"


def _reload_module(import_path: str):
    if not import_path:
        return

    module_path, _, _ = import_path.rpartition(".")
    module = importlib.import_module(module_path)
    importlib.reload(module)


class BaseSettings(_BaseSettings):
    @no_type_check
    def __setattr__(self, name, value):
        """
        Patch __setattr__ to be able to use property setters.
        From:
            https://github.com/pydantic/pydantic/issues/1577#issuecomment-790506164
        """
        try:
            super().__setattr__(name, value)
        except ValueError as e:
            setters = inspect.getmembers(
                self.__class__,
                predicate=lambda x: isinstance(x, property) and x.fset is not None,
            )
            for setter_name, func in setters:
                if setter_name == name:
                    object.__setattr__(self, name, value)
                    break
            else:
                raise e

    def dict(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().dict(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )

    def json(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().json(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )


class Settings(BaseSettings):
    class Config:
        env_file = ENV_FILE_SETTINGS
        env_prefix = ENV_PREFIX_SETTINGS

    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )

    parallel_workers: int = Field(
        default=DEFAULT_PARALLEL_WORKERS,
        description=f"Number of workers to run document-generation. "
                    f"Default is {DEFAULT_PARALLEL_WORKERS}."
    )

    parallel_workers_timeout: int = Field(
        default=5,
        description=f"Grace timeout to wait until the workers shut down when stopping DGServer."
    )

    server_name: str = Field(
        default="automl-server",
        description=f"Name of the server."
    )

    server_version: str = Field(
        default=__version__,
        description=f"Version of the server."
    )

    host: str = Field(
        default="0.0.0.0",
        description=f"Host where to listen for connections."
    )

    http_port: int = Field(
        default=8080,
        description=f"Port where to listen for HTTP / REST connections."
    )

    root_path: str = Field(
        default="",
        description=f"Set the ASGI root_path for applications submounted below a given URL path."
    )
    # MySql配置
    mysql_enabled: bool = Field(
        default=False,
        description=f"Enable MySql."
    )
    drivername_mysql: str = Field(
        default="mysql+mysqldb",
        description=f"Database driver name."
    )
    host_mysql: str = Field(
        default="124.70.188.119",
        description=f"Database host."
    )
    port_mysql: int = Field(
        default=3307,
        description=f"Database port."
    )
    username_mysql: str = Field(
        default="root",
        description=f"Database username."
    )
    password_mysql: str = Field(
        default="bdilab@1308",
        description=f"Database password."
    )
    database_mysql: str = Field(
        default="automl",
        description=f"Database name."
    )
    query_mysql: dict = Field(
        default={"charset": "utf8mb4", "collation": "utf8mb4_general_ci"},
        description=f"Database query."
    )
    async_enabled: bool = Field(
        default=False,
        description=f"Enable async mode."
    )
    # Kubernetes
    kubernetes_enabled: bool = Field(
        default=False,
        description=""
    )
    kube_config_file: str = Field(
        default=KUBE_CONFIG_FILE_PATh,
        description="Path to the kube-config file."
    )
    # TFJob
    base_image: str = Field(
        default='autotrain:0.0.7',
        description=''
    )
    namespcae: str = Field(
        default='zauto',
        description=''
    )
    # Model selection
    model_selection_enabled: bool = Field(
        default=False,
        description='Launch the model selection module'
    )
    prompt_template_file_path: str = Field(
        default=PROMPT_TEMPLATE_FILE_PATH,
        description="Model selection prompts template file path"
    )
    model_metadata_file_path: str = Field(
        default=MODEL_METADATA_FILE_PATH,
        description="Model metadata file path"
    )
    env_file_path: str = Field(
        default=ENV_FILE_PATH,
        description="Path to 'environment variables' file"
    )
    # OpenAI API
    llm_name_or_path: str = Field(
        # default="qwen1.5-14b-chat",
        default="gpt-3.5-turbo",
    )
    embedding_name_or_path: str = Field(
        # default="text-davinci-003",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    openai_api_key: str = Field(
        # default="EMPTY"
        default="sk-kUx6CDr4B0Elct2aq015N9tyu6siuSxsMrS0j4g86Ssf3w1l"
        # default="sk-no-key-required"
    )
    openai_api_base: str = Field(
        # default="http://10.78.180.40:7002/v1"
        default="https://api.chatanywhere.com.cn"
    )
    # Monitor
    monitor_enabled: bool = Field(
        default=False,
        description="Start the monitoring service"
    )
    host_info_file_path: str = Field(
        default=HOST_INFO_FILE_PATH,
        description="Host info file path"
    )