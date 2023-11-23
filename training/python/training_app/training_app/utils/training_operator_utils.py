# By DCJ, XDU
# 开发时间：2023/11/13 PM 11:45
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

#import training_command
from .training_command import command

DEFAULT_NAMESPACE = "zauto"


def _construct_v1_container(name: str, task_type: str, image: str):
    return V1Container(
        name=name,
        image=image,
        image_pull_policy="IfNotPresent",
        command=command[task_type.replace("-","_")],
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


def _construct_kubeflow_v1_tfjob(meta_name: str, meta_namespace: str, worker: V1Container, chief: V1Container,
                                 ps: V1Container) -> KubeflowOrgV1TFJob:
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
        training_project_task_type: str,
        image_full: str,
        model_path: str,
        data_path: str,
        output_path: str
) -> KubeflowOrgV1TFJob:
    """
        生成TrainingOperator自定义资源对象
        Args:
            training_project_name: 训练项目名称.
            training_project_task_type: 训练项目的任务类型.
            image_full: 训练基础镜像.
            model_dir: 模型文件路径.
            data_dir: 数据文件路径.
            output_dir: 训练结果输出路径.
    """
    container = _construct_v1_container(name=training_project_name, task_type=training_project_task_type, image=image_full)
    worker = _construct_v1_replica_spec(container=container, data_path=data_path, model_path=model_path,
                                        output_path=output_path)
    chief = _construct_v1_replica_spec(container=container, data_path=data_path, model_path=model_path,
                                       output_path=output_path)
    ps = _construct_v1_replica_spec(container=container, data_path=data_path, model_path=model_path,
                                    output_path=output_path)
    return _construct_kubeflow_v1_tfjob(
        meta_name=training_project_name,
        meta_namespace=DEFAULT_NAMESPACE,
        worker=worker,
        chief=chief,
        ps=ps
    )
