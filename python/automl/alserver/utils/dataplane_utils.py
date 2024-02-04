import os
import json
import shutil
from typing import Dict, Any
from pathlib import Path

TRAINING_PARAMETERS_FILE_NAME = 'traininig-parameters.json'
IMAGE_FOLDER_NAME = 'image'

WORKSPACE_DIR_IN_CONTAINER = '/metadata'
DATA_DIR_IN_CONTAINER = '/metadata/datasets'

EXCLUDE_ATTRIBUTES = [
    'model_type', 'task_type', 'trainer_class_name',
    'tp_project_name', 'tp_overwrite',  'tp_directory',
    'dp_feature_extractor_class_name'
]

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

def get_automl_metadata_base_dir():
    return os.path.join(PARENT_DIR, "metadata")

def generate_training_project_workspace_dir(worspace_name: str) -> str:
    workspace_dir = Path(os.path.join(get_automl_metadata_base_dir(), worspace_name))
    workspace_dir.mkdir(parents=True)
    return workspace_dir.__str__()

def get_training_project_data_dir_in_container():
    return DATA_DIR_IN_CONTAINER

def get_training_job_name(training_project_id, training_project_name):
    return '-'.join([str(training_project_name), str(training_project_id)])

def save_dict_to_json_file(data: Dict[str, Any], json_file: str):
    with open(json_file, "w") as json_file:
        json.dump(data, json_file)

def remove_workspace_dir(workspace_dir: str):
    if not Path(workspace_dir).exists():
        return
    
    shutil.rmtree(workspace_dir)
    
def get_training_params_dict(self, task_type: str, model_type: str):
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