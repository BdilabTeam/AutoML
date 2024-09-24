from typing import Optional, Union, List, Literal
import numpy as np

from ...utils.configuration_utils import BaseTrainerConfig

class DenseNetTrainerConfig(BaseTrainerConfig):
    model_type = "densenet"
    def __init__(
        self,
        task_type: str,
        trainer_class_name: str,
        # Data pipeline
        # AutoFeatureExtractor
        dp_enable_auto_feature_extract: bool = False,
        dp_feature_extractor_class_name = "GAForDenseNetFeatureExtractor",
        dp_feature_num: int = 2, 
        dp_svm_weight: float = 1.0, 
        dp_feature_weight: float = 0, 
        dp_C: Union[float, np.ndarray] = 1.0, 
        dp_keep_prob: float = 0.8, 
        dp_mutate_prob: float = 0.1, 
        dp_iters: int = 1,
        # Model pipeline
        mp_enable_categorical_to_numerical: Optional[bool] = True,
        # DenseBlock
        mp_num_layers: Optional[List[int]] = [1, 2, 3],
        mp_num_units: Optional[List[int]] = [16, 32, 64, 128, 256, 512, 1024],
        mp_use_batchnorm: Optional[bool] = True,
        mp_dropout: Optional[List[float]] = [0.25, 0.5],
        # ClassificationHead config
        mp_multi_label: bool = False,
        # Train pipeline
        # AutoModel
        tp_project_name: str = "auto_model",
        tp_directory: str = None,
        tp_max_trials: int = 1,
        tp_objective: str = None,
        tp_tuner: Literal["greedy", "bayesian", "hyperband", "random"] = "greedy",
        tp_overwrite: bool = True,
        tp_seed: Optional[int] = None,
        tp_max_model_size: Optional[int] = None,
        # AutoModel.fit()
        tp_batch_size: int = 8,
        tp_validation_split: float = 0.2,
        tp_epochs: Optional[int] = 100,
        **kwargs
    ) -> None:
        super().__init__(task_type=task_type, trainer_class_name=trainer_class_name, tp_project_name=tp_project_name, tp_directory=tp_directory)
        # AutoFeatureExtractor
        self.dp_enable_auto_feature_extract=dp_enable_auto_feature_extract
        self.dp_feature_num = dp_feature_num
        self.dp_svm_weight = dp_svm_weight
        self.dp_feature_weight = dp_feature_weight
        self.dp_C = dp_C
        self.dp_keep_prob = dp_keep_prob
        self.dp_mutate_prob = dp_mutate_prob
        self.dp_iters = dp_iters
        self.dp_feature_extractor_class_name = dp_feature_extractor_class_name

        self.mp_enable_categorical_to_numerical = mp_enable_categorical_to_numerical
        # DenseBlock
        self.mp_num_layers = mp_num_layers
        self.mp_num_units = mp_num_units
        self.mp_use_batchnorm = mp_use_batchnorm
        self.mp_dropout = mp_dropout
        # ClassificationHead config
        self.mp_multi_label=mp_multi_label
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
        self.tp_batch_size=tp_batch_size
        self.tp_epochs = tp_epochs
        self.tp_validation_split = tp_validation_split