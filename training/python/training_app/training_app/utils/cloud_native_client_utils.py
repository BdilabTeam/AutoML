import os
from dataclasses import dataclass, field
from typing import Optional

from kubeflow.katib import ApiClient
from kubernetes import client, config
from kubeflow import katib
from kubeflow.training import TrainingClient

def is_running_in_k8s():
    return os.path.isdir("/var/run/secrets/kubernetes.io/")

@dataclass
class MixinApiClient(object):
    """Cloud-Native Component Client
        Args:
            config_file: Path to the kube-config file. Defaults to ~/.kube/config.
            context: Set the active context. Defaults to current_context from the kube-config.
            client_configuration: Client configuration for cluster authentication.
                You have to provide valid configuration with Bearer token or
                with username and password.
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
            if self.config_file or not is_running_in_k8s():
                config.load_kube_config(config_file=self.config_file, context=self.context)
            else:
                config.load_incluster_config()
                self.in_cluster = True
        # k8s客户端
        k8s_client = client.ApiClient(self.client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)
        self.api_client = client.ApiClient()
       

def _get_mixin_api_client(config_file: Optional[str], context: Optional[str], client_configuration: Optional[client.Configuration]) -> ApiClient:
    return MixinApiClient(config_file=config_file, context=context, client_configuration=client_configuration)


def _construct_configuration(self, host="http://localhost",
                             api_key=None, api_key_prefix=None,
                             username=None, password=None,
                             discard_unknown_keys=False,) -> client.Configuration:
    return client.Configuration(host=host, api_key=api_key,
                                api_key_prefix=api_key_prefix, 
                                username=username, password=password,
                                discard_unknown_keys=discard_unknown_keys)
    
    
def _get_katib_client(config_file: Optional[str], context: Optional[str], client_configuration: Optional[client.Configuration]) -> katib.KatibClient:
    """katib client
        Args:
            config_file: Path to the kube-config file. Defaults to ~/.kube/config.
            context: Set the active context. Defaults to current_context from the kube-config.
            client_configuration: Client configuration for cluster authentication.
                You have to provide valid configuration with Bearer token or
                with username and password.
    """
    return katib.KatibClient(config_file=config_file, context=context, client_configuration=client_configuration)


def _get_kubeflow_trainig_operator_client(config_file: Optional[str], context: Optional[str], client_configuration: Optional[client.Configuration]) -> katib.KatibClient:
    """kubeflow training operator client
        Args:
            config_file: Path to the kube-config file. Defaults to ~/.kube/config.
            context: Set the active context. Defaults to current_context from the kube-config.
            client_configuration: Client configuration for cluster authentication.
                You have to provide valid configuration with Bearer token or
                with username and password.
    """
    return TrainingClient(config_file=config_file, context=context, client_configuration=client_configuration)