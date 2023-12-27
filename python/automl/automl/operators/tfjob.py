import json
from typing import Optional, List

from kubernetes.client import (
    V1PodTemplateSpec,
    V1ObjectMeta,
    V1PodSpec,
    V1Container,
    V1VolumeMount,
    V1Volume,
    V1EnvVar,
    V1ResourceRequirements,
    V1JobCondition
)

from kubeflow.training import (
    KubeflowOrgV1TFJob,
    KubeflowOrgV1TFJobSpec,
    V1RunPolicy,
    V1ReplicaSpec
)

from .settings import TFJobSettings, KubeConfigSettings
from ..utils import MixinCloudNativeClient
from .errors import (
    CreateTFJobError,
    GetTFJobError,
    GetTFJobConditionsError,
    GetTFJobLogsError,
    ValueError
)
from .logging import get_logger


logger = get_logger(__name__)

def _construct_v1_env_var(settings: TFJobSettings):
    env_vars = []
    if settings.envs:
        for k, v in settings.envs.items():
            env_vars.append(V1EnvVar(k, v))
    return env_vars

def _construct_v1_volume_mounts(settings: TFJobSettings):
    volume_mounts = []
    if settings.volume_mapping:
        for volume_name, path_mapping in settings.volume_mapping.items():
            for _, internal_path in path_mapping.items():
                volume_mounts.append(V1VolumeMount(name=volume_name, mount_path=internal_path))
    return volume_mounts

def _construct_v1_volumes(settings: TFJobSettings):
    volumes = []
    if settings.volume_mapping:
        for volume_name, path_mapping in settings.volume_mapping.items():
            for external_path, _ in path_mapping.items():
                volumes.append(V1Volume(name=volume_name, host_path=external_path))
    return volumes

def _construct_args(settings: TFJobSettings):
    args = []
    if settings.args:
        for arg_name, arg_value in settings.args.items():
            args.append(arg_name)
            args.append(json.dumps(arg_value))
    return args

def _construct_tf_replica_spec(settings: TFJobSettings) -> V1ReplicaSpec:
    return V1ReplicaSpec(
        replicas=settings.replicas,
        restart_policy=settings.restart_policy,
        template=V1PodTemplateSpec(
            metadata=V1ObjectMeta(
                annotations=settings.annotations
            ),
            spec=V1PodSpec(
                node_name=settings.nodeName,
                containers=[
                    V1Container(
                        name=settings.container_name,
                        image=settings.image,
                        image_pull_policy=settings.image_pull_policy,
                        command=settings.command,
                        args=_construct_args(settings=settings),
                        resources=V1ResourceRequirements(
                            requests=settings.request,
                            limits=settings.limit
                        ),
                        env=_construct_v1_env_var(settings=settings),
                        volume_mounts=_construct_v1_volume_mounts(settings=settings)
                    )
                ],
                volumes=_construct_v1_volumes(settings=settings)
            )
        )
    )

def _construct_kubeflow_v1_tfjob(settings: TFJobSettings):
    return KubeflowOrgV1TFJob(
        api_version=settings.api_version,
        kind=settings.kind,
        metadata=V1ObjectMeta(name=settings.name, namespace=settings.namespace),
        spec=KubeflowOrgV1TFJobSpec(
            run_policy=V1RunPolicy(clean_pod_policy="None"),
            tf_replica_specs={
                "PS": _construct_tf_replica_spec(settings=settings),
                "Worker": _construct_tf_replica_spec(settings=settings),
                # "Chief": _construct_tf_replica_spec(settings=settings),
            }
        )
    )

class TFJobOrchestrator(object):
    def __init__(self, settings: KubeConfigSettings):
        self.training_operator_client = MixinCloudNativeClient(
            config_file=settings.kube_config_file, 
            context=settings.context,
            client_configuration=settings.client_configuration
        ).kubeflow_trainig_operator_client
        
    def construct_kubeflow_v1_tfjob(self, settings: TFJobSettings):
        return _construct_kubeflow_v1_tfjob(settings=settings)
    
    def create_kubeflow_v1_tfjob(
        self,
        tfjob: KubeflowOrgV1TFJob,
        namespace: Optional[str] = None
    ):
        if not tfjob:
            raise ValueError(msg="The 'tfjob' field can not be null")
        try:
            if not namespace:
                self.training_operator_client.create_tfjob(tfjob=tfjob)
            else:
                self.training_operator_client.create_tfjob(tfjob=tfjob, namespace=namespace)
        except:
            raise CreateTFJobError(msg="Create TFJob failed")
        else:
            logger.info(f"Create tfjob successfully, The following is the resource representation created\n{self.tfjob}")

    def get_kubeflow_v1_tfjob(
        self,
        name: str, 
        namespace: Optional[str] = None
    ) -> KubeflowOrgV1TFJob:
        if not name:
            raise ValueError(msg="The 'name' field can not be null")
        try:
            if not namespace:
                return self.training_operator_client.get_tfjob(name=name)
            else:
                return self.training_operator_client.get_tfjob(name=name, namespace=namespace)
        except:
            raise GetTFJobError("Failed to get tfjob")
            
    
    def delete_kubeflow_v1_tfjob(
        self,
        name: str,
        namespace: Optional[str] = None
    ):
        if not name:
            raise ValueError(msg="The 'name' field can not be null")
        try:
            if not namespace:
                self.training_operator_client.delete_tfjob(name=name)
            else:
                self.training_operator_client.delete_tfjob(name=name, namespace=namespace)
        except:
            raise
        else:
            logger.info("Delete tfjob successfully")
    
    def get_kubeflow_v1_tfjob_conditions(
        self,
        name: str, 
        namespace: Optional[str] = None,
    ):
        if not name:
            raise ValueError(msg="The 'name' field can not be null")
        try:
            if not namespace:
                return self.training_operator_client.get_job_conditions(name=name)
            else:
                return self.training_operator_client.get_job_conditions(name=name, namespace=namespace)
        except:
            raise GetTFJobConditionsError("Fail to get the conditions of the tfjob")
    
    def get_kubeflow_v1_tfjob_logs(
        self,
        name: str, 
        namespace: Optional[str] = None,
    ):
        if not name:
            raise ValueError(msg="The 'name' field can not be null")
        try:
            if not namespace:
                return self.training_operator_client.get_job_logs(name=name)
            else:
                return self.training_operator_client.get_job_logs(name=name, namespace=namespace)
        except:
            raise GetTFJobLogsError("Fail to get the logs of the tfjob")