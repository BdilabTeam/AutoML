from typing import List, Optional, Tuple, Literal

from ...utils.configuration_utils import (
    BaseTrainerConfig
)


class ConvNetTrainerConfig(BaseTrainerConfig):
    model_type = "convnet"
    def __init__(
        self,
        task_type: str,
        trainer_class_name: str,
        # Data Pipeline
        dp_batch_size: Optional[int] = 16,
        dp_color_mode: Optional[str] = None,
        dp_image_size: Optional[Tuple[float, float]] = None,
        dp_interpolation: Optional[str] = None,
        dp_shuffle: Optional[bool] = None,
        dp_seed: Optional[int] = None,
        dp_validation_split: Optional[float] = None,
        dp_subset: Optional[str] = None,
        # Model Pipeline
        # Normalization
        mp_enable_normalization: bool = True,
        # ImageAugmentation
        mp_enable_image_augmentation: bool = True,
        mp_translation_factor: Optional[List[float]] = None,
        mp_vertical_flip: Optional[bool] = None,
        mp_horizontal_flip: Optional[bool] = None,
        mp_rotation_factor: Optional[List[float]] = None,
        mp_zoom_factor: Optional[List[float]] = None,
        mp_contrast_factor: Optional[List[float]] = None,
        # ConvNets
        mp_kernel_size: Optional[List[int]] = [3, 5, 7],
        mp_num_blocks: Optional[List[int]] = [1, 2, 3],
        mp_num_layers: Optional[List[int]] = [1, 2],
        mp_filters: Optional[List[int]] = [16, 32, 64, 128, 256, 512],
        mp_max_pooling: Optional[bool] = True,
        mp_separable: Optional[bool] = None,
        mp_dropout: Optional[List[float]] = [0.25, 0.5],
        # Train pipeline
        # AutoModel
        tp_project_name: str = "auto_model",
        tp_max_trials: int = 1,
        tp_directory: str = None,
        tp_objective: str = "val_loss",
        tp_tuner: Literal["greedy", "bayesian", "hyperband", "random"] = "greedy",
        tp_overwrite: bool = False,
        tp_seed: Optional[int] = None,
        tp_max_model_size: Optional[int] = None,
        # AutoModel.fit()
        tp_batch_size: int = 8,
        tp_epochs: Optional[int] = 100,
        tp_validation_split: float = 0.2,
    ):
        super().__init__(task_type=task_type, trainer_class_name=trainer_class_name, tp_project_name=tp_project_name, tp_directory=tp_directory)
        # Data Pipeline
        self.dp_batch_size = dp_batch_size
        self.dp_color_mode = dp_color_mode
        self.dp_image_size = dp_image_size
        self.dp_interpolation = dp_interpolation
        self.dp_shuffle = dp_shuffle
        self.dp_seed = dp_seed
        self.dp_validation_split = dp_validation_split
        self.dp_subset = dp_subset
        # Normalization
        self.mp_enable_normalization = mp_enable_normalization
        # ImageAugmentation
        self.mp_enable_image_augmentation = mp_enable_image_augmentation
        self.mp_translation_factor = mp_translation_factor
        self.mp_vertical_flip = mp_vertical_flip
        self.mp_horizontal_flip = mp_horizontal_flip
        self.mp_rotation_factor = mp_rotation_factor
        self.mp_zoom_factor = mp_zoom_factor
        self.mp_contrast_factor = mp_contrast_factor
        # ConvNets
        self.mp_kernel_size = mp_kernel_size
        self.mp_num_blocks = mp_num_blocks
        self.mp_num_layers = mp_num_layers
        self.mp_filters = mp_filters
        self.mp_max_pooling = mp_max_pooling
        self.mp_separable = mp_separable
        self.mp_dropout = mp_dropout
        # AutoModel
        self.tp_project_name = tp_project_name
        self.tp_max_trials = tp_max_trials
        self.tp_directory = tp_directory
        self.tp_objective = tp_objective
        self.tp_tuner = tp_tuner
        self.tp_overwrite = tp_overwrite
        self.tp_seed = tp_seed
        self.tp_max_model_size = tp_max_model_size
        # AutoModel.fit()
        self.tp_batch_size = tp_batch_size
        self.tp_epochs = tp_epochs
        self.tp_validation_split = tp_validation_split