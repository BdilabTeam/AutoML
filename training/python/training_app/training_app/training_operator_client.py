from kubernetes import client, config, watch

from kubeflow.training import TrainingClient

from kubeflow.training.api_client import ApiClient

from kubeflow.training.utils import utils

from utils import logging, training_command

from dataclasses import (
    dataclass,
    field
)
from typing import(
    Optional
)

# import training_app.utils.training_operator_utils as training_operator_utils
from utils import training_operator_utils

logger = logging.get_logger(__name__)

namespace = training_operator_utils.DEFAULT_NAMESPACE

@dataclass
class TrainingOperatorClient():
    """
        TrainingOperatorClient constructor .
    """
    
    config_file: Optional[str] = field(
        default=None,
        metadata={
            help: "Path to the kube-config file. Defaults to ~/.kube/config."
        }
    )
    context: Optional[str] = field(
        default=None,
        metadata={
            help: "Set the active context. Defaults to current_context from the kube-config."
        }
    )
    client_configuration: Optional[client.Configuration] = field(
        default=None,
        metadata={
            help: "Client configuration for cluster authentication."
        }
    )
    
    
    def __post_init__(self):
        """
            dataclass自动生成的__init__方法之后被调用的一个方法.
        """
        # If client configuration is not set, use kube-config to access Kubernetes APIs.
        if self.client_configuration is None:
            # Load kube-config or in-cluster config.
            if self.config_file or not utils.is_running_in_k8s():
                config.load_kube_config(config_file=self.config_file, context=self.context)
            else:
                config.load_incluster_config()

        k8s_client = client.ApiClient(self.client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)
        self.api_client = ApiClient()
        # Load kubeflow training client
        self.training_client = TrainingClient(config_file=self.config_file)
        # client = TrainingClient(config_file="/root/workspace/YJX/Env/yjx/app/public/config")


    def create_training_job(
        self,
        training_project_name: str,
        training_project_task_type: str,
        model_path: str,
        data_path: str,
        output_path: str,
        image_full: str = "registry.cn-hangzhou.aliyuncs.com/treasures/training-script-env:v0.0.3",
        namespace: str = namespace,
        job_kind: Optional[str] = None,
    ):
        # 生成training-operator
        tfjob = training_operator_utils._generate_training_operator(
            self=self,
            training_project_name=training_project_name,
            training_project_task_type=training_project_task_type,
            image_full=image_full,
            model_path=model_path,
            data_path=data_path,
            output_path=output_path
        )
        try:
            self.training_client.create_tfjob(tfjob=tfjob, namespace=namespace)
        except:
            raise 
        else:
            logger.info(f"Create successfully, The following is the resource representation created\n{tfjob}")


    def get_training_job(
        self,
        name: str, 
        namespace: str = namespace,
        job_kind: Optional[str] = None
    ):
        try:
            return self.training_client.get_tfjob(name=name, namespace=namespace)
        except:
            raise


    # 获取 Job 状态
    def get_training_job_conditions(
        self,
        name: str, 
        namespace: str = namespace,
        job_kind: Optional[str] = None
    ):
        try:
            return self.training_client.get_job_conditions(name=name, namespace=namespace)
        except:
            raise


    # 等待特定 Job 完成


    # 获取训练日志
    def get_training_job_logs(
        self,
        name: str, 
        namespace: str = namespace,
        job_kind: Optional[str] = None
    ):
        try:
            return self.training_client.get_job_logs(name=name, namespace=namespace)
        except:
            raise

    # 删除 
    def delete_training_job(
        self,
        name: str, namespace: str,
        job_kind: Optional[str] = None
    ):
        try:
            self.training_client.delete_tfjob(name=name, namespace=namespace)
        except:
            raise
        else:
            logger.info("delete successfully")