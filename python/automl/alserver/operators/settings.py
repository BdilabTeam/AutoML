from pydantic import Field
from typing import Dict, Optional, List, Any
from kubernetes import client

from ..settings import BaseSettings

class KubeConfigSettings(BaseSettings):
    kube_config_file: Optional[str] = Field(
        default="~/.kube/config",
        description="Path to the kube-config file."
    )
    context: Optional[str] = Field(
        default=None,
        description="The active context. Defaults to current_context from the kube-config."
    )
    client_configuration: Optional[client.Configuration] = Field(
        default=None,
        description="Client configuration for cluster authentication"
    )

class JobSettings(BaseSettings):
    api_version: str = Field(
        default="kubeflow.org/v1",
    )
    kind: str = Field(
        default="TFJob"
    )

class TFJobSettings(JobSettings):
    name: str = Field(
        default=None,
        description="The name of the tfjob"
    )
    namespace: str = Field(
        default="zauto",
        description="Namespace for deploying tfjob"
    )
    annotations: Dict[str, str] = Field(
        default={
            "ALIYUN_COM_GPU_MEM_ASSIGNED": "false",
            "ALIYUN_COM_GPU_MEM_DEV": "48",
            "ALIYUN_COM_GPU_MEM_POD": "2",
            "ALIYUN_COM_GPU_MEM_IDX": "",
            "sidecar.istio.io/inject": "false"
        },
    )
    # spec
    nodeName: str = Field(
        default=None,
        description="Name of the node on which the tfjob will be run"
    )
    replicas: int = Field(
        default=1,
        description="Number of tfjob copies"
    )
    restart_policy: str = Field(
        default="OnFailure",
        description="tfjob restart strategy"
    )
    # container
    contrainer_id: str = Field(
        default="training",
        description="The name of the container"
    )
    image: str = Field(
        default="autotrain:latest",
        description="The full name (name:tag) of the image"
    )
    image_pull_policy: str = Field(
        default="IfNotPresent",
        description="Mirror pulling strategies include: IfNotPresent、Always、Never."
    )
    request: Dict[str, str] = Field(
        default={
            "nvidia.com/gpu": "1"
        },
        description="Amount of resources requested by tfjob"
    )
    limit: Dict[str, str] = Field(
        default={
            "nvidia.com/gpu": "1"
        },
        description="The upper limit of resources used by tfjob"
    )
    envs: Dict[str, str] = Field(
        default={
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_DEBUG": "INFO"
        },
        description="Define environment variables within the container"
    )
    volume_mapping: Dict[str, Dict[str, str]] = Field(
        default={
            "data-dir": {
                "": "/treasures/data"
            },
            "output-dir": {
                "": "/treasures/output"
            }
        },
        description="key: vlume_name; value: external_path:internal_path"
    )
    command: List[str] = Field(
        default=[
            "python", 
            "run_densenet.py"
        ]
    )
    args: Dict[str, Dict[str, Any]] = Field(
        default={
            "--args_dict": {
                "task_type": "structured_data_classification",
                "model_type": "densenet",
                "train_dir": "/root/workspace/YJX/auto-ml/automl/python/autotrain/autotrain/datasets/train.csv",
                "output_dir": "/root/workspace/YJX/auto-ml/automl/python/autotrain/tests/output",
                "tp_overwrite": True,
                "tp_project_name": "test",
                "tp_max_trials": 3,
                "tp_objective": "val_loss",
                "tp_tuner": "greedy",
                "tp_batch_size": 32,
                "epochs": 5,
                "validation_split": 0.3,
                "is_early_stop": False,
                "do_auto_feature_extract": False,
                "do_auto_hyperparameter_tuning": True,
                "num_layers": [1, 2, 3],
                "num_units": [16, 32, 64, 128, 256, 512, 1024],
                "use_batchnorm": True,
                "dropout": [0.0, 0.25, 0.5],
                "iters": 2
            }
        }
    )
    
    