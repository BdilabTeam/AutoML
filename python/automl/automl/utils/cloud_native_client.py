import os
from dataclasses import dataclass, field
from typing import Optional
from kubernetes import client, config
from kubeflow.training import TrainingClient

def is_running_in_k8s():
    return os.path.isdir("/var/run/secrets/kubernetes.io/")

@dataclass
class MixinCloudNativeClient(object):
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
            k8s_client = client.ApiClient(self.client_configuration)
            self._custom_api = client.CustomObjectsApi(k8s_client)
            self._core_api = client.CoreV1Api(k8s_client)
            self._api_client = client.ApiClient()
            self._kubeflow_trainig_operator_client = TrainingClient(config_file=self.config_file, context=self.context, client_configuration=self.client_configuration)
            
    @property
    def custom_api(self):
        return self._custom_api

    @property
    def core_api(self):
        return self._core_api
    
    @property
    def api_client(self):
        return self._api_client
    
    @property
    def kubeflow_trainig_operator_client(self):
        return self._kubeflow_trainig_operator_client
      