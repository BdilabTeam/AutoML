import concurrent.futures
import os
import json
import shutil
from typing import Dict, Any
from pathlib import Path
import concurrent

EXPERIMENT_SUMMARY_FILE_NAME = 'summary.json'
EXPERIMENT_TRAINING_PARAMETERS_FILE_NAME = 'traininig-parameters.json'
DATASETS_FOLDER_NAME = 'datasets'
IMAGE_FOLDER_NAME = 'image'
BEST_MODEL_FOLDER_NAME = 'best_model'
TP_PROJECT_NAME = 'output'

WORKSPACE_DIR_IN_CONTAINER = '/metadata'
DATA_DIR_IN_CONTAINER = '/metadata/datasets'

EXCLUDE_ATTRIBUTES = [
    'model_type', 'task_type', 'trainer_class_name',
    'tp_project_name', 'tp_overwrite',  'tp_directory',
    'tp_tuner', 'dp_feature_extractor_class_name'
]

BUCKET_NAME = "automl"

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

def get_automl_metadata_base_dir():
    return os.path.join(PARENT_DIR, "metadata")

def generate_experiment_workspace_dir(experiment_name: str) -> str:
    workspace_dir = Path(os.path.join(get_automl_metadata_base_dir(), experiment_name))
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir.__str__()

def get_experiment_workspace_dir(experiment_name: str) -> str:
    return os.path.join(get_automl_metadata_base_dir(), experiment_name)

def get_experiment_output_dir(experiment_name: str):
    return os.path.join(get_experiment_workspace_dir(experiment_name), TP_PROJECT_NAME)

def get_experiment_summary_file_path(experiment_name: str):
    return os.path.join(get_experiment_workspace_dir(experiment_name), TP_PROJECT_NAME, EXPERIMENT_SUMMARY_FILE_NAME)

def get_experiment_summary_file_url(experiment_name: str):
    return os.path.join('/api/v1/metadata', experiment_name, TP_PROJECT_NAME, EXPERIMENT_SUMMARY_FILE_NAME)
    
def get_experiment_data_dir_in_container():
    return DATA_DIR_IN_CONTAINER

def save_dict_to_json_file(data: Dict[str, Any], json_file: str):
    with open(json_file, "w") as json_file:
        json.dump(data, json_file)

def remove_experiment_workspace_dir(experiment_name: str):
    experiment_workspace_dir = get_experiment_workspace_dir(experiment_name=experiment_name)
    if not Path(experiment_workspace_dir).exists():
        return
    
    shutil.rmtree(experiment_workspace_dir)
    
def get_training_params_dict(task_type: str, model_type: str):
    """Get the configuration parameters of the trainer"""
    from autotrain import AutoConfig

    trainer_id = task_type + '/' + model_type
    config = AutoConfig.from_repository(trainer_id=str.lower(trainer_id))
    
    config_dict = config.__dict__
    training_params_dict = {}
    for key, value in config_dict.items():
        if key in EXCLUDE_ATTRIBUTES:
            continue
        training_params_dict[key] = value
    return training_params_dict

def get_experiment_data_dir(experiment_name: str):
    return os.path.join(get_experiment_workspace_dir(experiment_name), DATASETS_FOLDER_NAME)

def get_experiment_training_params_file_path(experiment_name: str):
    return os.path.join(get_experiment_workspace_dir(experiment_name), EXPERIMENT_TRAINING_PARAMETERS_FILE_NAME)

def get_experiment_best_model_dir(experiment_name: str):
    return os.path.join(get_experiment_workspace_dir(experiment_name), TP_PROJECT_NAME, BEST_MODEL_FOLDER_NAME)

def get_yolo_experiment_best_model_dir(experiment_name: str):
    return os.path.join(get_experiment_workspace_dir(experiment_name), TP_PROJECT_NAME, 'weights')

def get_external_workspace_dir(experiment_name: str):
    return get_experiment_workspace_dir(experiment_name=experiment_name)   # 适用于本地部署
    # return "/nfs/automl/workspace/python" + get_experiment_workspace_dir(experiment_name=experiment_name)   # 适用于通过k8s部署
    
def get_server_host(ip, port):
    return "http://" + str(ip) + ":" + str(port)

def upload_dir_to_minio(minio_client, bucket_name, dir_path, prefix=''):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 构造 MinIO 中的路径
            minio_path = os.path.join(prefix, root.replace(dir_path, '')[1:], file)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future = executor.submit(
                    minio_client.fput_object,
                    bucket_name=bucket_name,
                    object_name=minio_path,
                    file_path=os.path.join(root, file), 
                )
                future.add_done_callback(lambda _: print(f"上传成功: {minio_path}"))
            # try:
            #     # 上传文件到 MinIO
            #     client.fput_object(
            #         bucket_name=bucket_name,
            #         object_name=minio_path,
            #         file_path=os.path.join(root, file),
            #     )
            #     print(f"上传成功: {minio_path}")
            # except Exception as exc:
            #     print(f"上传失败: {minio_path}, 错误信息: {exc}")

def delete_dir_from_minio(minio_client, bucket_name, prefix = ''):
    from minio.deleteobjects import DeleteObject
    
    delete_object_list = list(
        map(
            lambda x: DeleteObject(x.object_name),
            minio_client.list_objects(
                BUCKET_NAME,
                prefix,
                recursive=True,
            ),
        )
    )
    errors = minio_client.remove_objects(BUCKET_NAME, delete_object_list)
    for error in errors:
        print("error occurred when deleting object", error)