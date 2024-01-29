from enum import Enum

class TaskType(Enum):
    STRUCTURED_DATA_CLASSIFICATION = "structured-data-classification"
    STRUCTURED_DATA_REGRESSION = "structured-data-regression"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_REGRESSION = "image-regression"

class ModelType(Enum):
    DENSENET = "densenet"
    RESNET = "resnet"

