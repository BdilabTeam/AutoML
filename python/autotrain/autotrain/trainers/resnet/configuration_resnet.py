from typing import List, Optional

from ..utils import TaskType

class ResNetConfig:
    model_type = "resnet"
    def __init__(
        self,
        task_type: str = "image_classification",
        # Normalization
        enable_normalization: bool = False,
        # ImageAugmentation
        enable_image_augmentation: bool = False,
        translation_factor: Optional[List[float]] = None,
        vertical_flip: Optional[bool] = None,
        horizontal_flip: Optional[bool] = None,
        rotation_factor: Optional[List[float]] = None,
        zoom_factor: Optional[List[float]] = None,
        contrast_factor: Optional[List[float]] = None,
        # ResNet
        version: Optional[str] = None,
        pretrained: Optional[bool] = None,
        # AutoModel config
        project_name: str = "auto_model",
        max_trials: int = 1,
        directory: Optional[str] = None,
        objective: str = "val_loss",
        tuner: str = "greedy",
        overwrite: bool = False,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        # AutoModel.fit() config
        batch_size: int = 32,
        epochs: Optional[int] = None,
        validation_split: float = 0.2,
        is_early_stop: Optional[bool] = True,
        
        
    ):
        self.task_type = task_type
        # Normalization
        self.enable_normalization = enable_normalization
        # ImageAugmentation
        self.enable_image_augmentation = enable_image_augmentation
        self.translation_factor = translation_factor
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.contrast_factor = contrast_factor
        # ResNet
        self.version = version
        self.pretrained = pretrained
        # AutoModel
        self.project_name = project_name
        self.max_trials = max_trials
        self.directory = directory
        self.objective = objective
        self.tuner = tuner
        self.overwrite = overwrite
        self.seed = seed
        self.max_model_size = max_model_size
        # AutoModel.fit()
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.is_early_stop = is_early_stop
        #TODO 修改映射方法
        if task_type == TaskType.IMAGE_CLASSIFICATION.value:
            self.trainer_class_name = "AKResNetForImageClassificationTrainer"
        elif task_type == TaskType.IMAGE_REGRESSION.value:
            self.trainer_class_name = "AKResNetForImageRegressionTrainer"
        else:
            raise ValueError(f"The model type '{ResNetConfig.model_type}' does not support the task type '{task_type}'")
        