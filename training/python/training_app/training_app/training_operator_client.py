from kubernetes import client, config, watch

from kubernetes.client import (
    V1PodTemplateSpec,
    V1ObjectMeta,
    V1PodSpec,
    V1Container,
    V1VolumeMount,
    V1Volume,
    V1HostPathVolumeSource,
    V1EnvVar
)

from kubeflow.training import (
    constants,
    KubeflowOrgV1TFJob,
    KubeflowOrgV1TFJobSpec,
    V1RunPolicy,
    TrainingClient,
    Configuration,
    V1ReplicaSpec
)
from kubeflow.training.api_client import ApiClient

from kubeflow.training.utils import utils

from utils import logging

from dataclasses import (
    dataclass,
    field
)
from typing import(
    Optional
)

logger = logging.get_logger(__name__)
namespace = "zauto"
# namespace = utils.get_default_target_namespace()

# 后续考虑移至utils包下
def _construct_v1_container(name: str, image: str):
    return V1Container(
        name=name,
        image=image,
        image_pull_policy="IfNotPresent",
        command=[
            "python",
            "run_image_classification.py",
            "--model_name_or_path=/treasures/model",
            # f"--dataset_name={data_name_or_path}",
            "--train_dir=/treasures/data",
            "--output_dir=/treasures/output/",
            "--remove_unused_columns=False",
            "--do_train",
            "--do_eval",
            "--learning_rate=2e-5",
            "--num_train_epochs=5",
            "--per_device_train_batch_size=8",
            "--per_device_eval_batch_size=8",
            "--logging_strategy=steps",
            "--logging_steps=10",
            "--evaluation_strategy=epoch",
            "--save_strategy=epoch",
            "--load_best_model_at_end=True",
            "--save_total_limit=3",
            "--seed=1337",
            "--ignore_mismatched_sizes=True"
        ],
        env=[
            V1EnvVar(name="TRANSFORMERS_OFFLINE", value="1"),
            V1EnvVar(name="HF_DATASETS_OFFLINE", value="1")
        ],
        volume_mounts=[
            V1VolumeMount(name="data-path", mount_path="/treasures/data"),
            V1VolumeMount(name="model-path", mount_path="/treasures/model"),
            V1VolumeMount(name="output-path", mount_path="/treasures/output"),
        ]
    )
    

def _construct_v1_replica_spec(container: V1Container,
                               data_path: str,
                               model_path: str,
                               output_path: str) -> V1ReplicaSpec:
    return V1ReplicaSpec(
        replicas=1,
        restart_policy="OnFailure",
        template=V1PodTemplateSpec(
            metadata=V1ObjectMeta(
                annotations={"sidecar.istio.io/inject": "false"}
            ),
            spec=V1PodSpec(
                node_name="node1",
                containers=[container],
                volumes=[
                    V1Volume(name="data-path", host_path=V1HostPathVolumeSource(path=data_path)),
                    V1Volume(name="model-path", host_path=V1HostPathVolumeSource(path=model_path)),
                    V1Volume(name="output-path", host_path=V1HostPathVolumeSource(path=output_path))
                ]
            )
        )
    )
    

def _construct_kubeflow_v1_tfjob(meta_name: str, meta_namespace: str, worker: V1Container, chief: V1Container, ps: V1Container) -> KubeflowOrgV1TFJob:
    return KubeflowOrgV1TFJob(
        api_version="kubeflow.org/v1",
        kind="TFJob",
        metadata=V1ObjectMeta(name=meta_name, namespace=meta_namespace),
        spec=KubeflowOrgV1TFJobSpec(
            run_policy=V1RunPolicy(clean_pod_policy="None"),
            tf_replica_specs={
                "Worker": worker,
                # "Chief": chief,
                "ps": ps
            }
        )
    )


def _generate_training_operator(
    self,
    training_project_name: str, 
    image_full: str,
    model_path: str,
    data_path: str,
    output_path: str
) -> KubeflowOrgV1TFJob:
    """
        生成TrainingOperator自定义资源对象
        Args:
            training_project_name: 训练项目名称.
            image_full: 训练基础镜像.
            model_dir: 模型文件路径.
            data_dir: 数据文件路径.
            output_dir: 训练结果输出路径.
    """
    container = _construct_v1_container(name=training_project_name, image=image_full)
    worker = _construct_v1_replica_spec(container=container, data_path=data_path, model_path=model_path, output_path=output_path)
    chief = _construct_v1_replica_spec(container=container, data_path=data_path, model_path=model_path, output_path=output_path)
    ps = _construct_v1_replica_spec(container=container, data_path=data_path, model_path=model_path, output_path=output_path)
    return _construct_kubeflow_v1_tfjob(
        meta_name=training_project_name,
        meta_namespace=namespace,
        worker=worker,
        chief=chief,
        ps=ps
    )


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
        model_path: str,
        data_path: str,
        output_path: str,
        image_full: str = "treasures/training:latest",
        namespace: str = namespace,
        job_kind: Optional[str] = None,
    ):
        # 生成training-operator
        tfjob = _generate_training_operator(
            self=self,
            training_project_name=training_project_name,
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
            self.training_client.delete_tfjob(name="mnist", namespace=namespace)
        except:
            raise
        else:
            logger.info("delete successfully")