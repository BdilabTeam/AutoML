from enum import Enum

SUPPORTED_TASK_TYPE = [
    "structured_data_classification",
    "structured_data_regression"
]

SUPPORTED_MODEL_TYPE = [
    "densenet"
]

class TaskType(Enum):
    STRUCTURED_DATA_CLASSIFICATION = "structured_data_classification"
    STRUCTURED_DATA_REGRESSION = "structured_data_regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_REGRESSION = "image_regression"
