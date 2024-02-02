import os

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

def generate_training_project_base_dir(training_project_id: int) -> str:
    return os.path.join(PARENT_DIR, "metadata", training_project_id)

def generate_training_project_output_dir(training_project_id: int) -> str:
    return os.path.join(PARENT_DIR, "metadata", training_project_id, "output")

def generate_training_project_data_dir(training_project_id: int) -> str:
    return os.path.join(PARENT_DIR, "metadata", training_project_id, "data")