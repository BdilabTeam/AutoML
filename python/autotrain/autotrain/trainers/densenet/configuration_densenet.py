from typing import Optional, Union, List
import numpy as np

from ..utils import TaskType

class DenseNetConfig():
    model_type = "densenet"
    def __init__(
        self,
        task_type: str = "structured_data_regression",
        # AutoModel config
        project_name: str = "auto_model",
        max_trials: int = 1,
        objective: str = "val_loss",
        tuner: str = "greedy",
        overwrite: bool = True,
        directory: Optional[str] = None,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        # AutoModel.fit() config
        batch_size: int = 32,
        validation_split: float = 0.2,
        epochs: Optional[int] = None,
        is_early_stop: Optional[bool] = True,
        enable_categorical_to_numerical: Optional[bool] = True,
        # DenseBlock config
        num_layers: Optional[List[int]] = None,
        num_units: Optional[List[int]] = None,
        use_batchnorm: Optional[bool] = True,
        dropout: Optional[List[float]] = None,
        # ClassificationHead config
        multi_label: bool = False,
        # AutoFeatureExtractor config
        do_auto_feature_extract: bool = False,
        feature_num: int = 2, 
        svm_weight: float = 1.0, 
        feature_weight: float = 0, 
        C: Union[float, np.ndarray] = 1.0, 
        keep_prob: float = 0.8, 
        mutate_prob: float = 0.1, 
        iters: int = 1,
        feature_extractor_class_name = "DenseNetFeatureExtractor",
        **kwargs
    ) -> None:
        self.task_type = task_type
        # AutoModel
        self.project_name = project_name
        self.max_trials = max_trials
        self.directory = directory
        self.objective = objective
        self.tuner = tuner
        self.overwrite = overwrite
        self.seed = seed
        self.max_model_size = max_model_size
        
        self.enable_categorical_to_numerical = enable_categorical_to_numerical
        # DenseBlock
        self.num_layers = num_layers
        self.num_units = num_units
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        # ClassificationHead config
        self.multi_label=multi_label
        # AutoModel.fit()
        self.batch_size=batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.is_early_stop = is_early_stop
        # AutoFeatureExtractor
        self.do_auto_feature_extract=do_auto_feature_extract
        self.feature_num = feature_num
        self.svm_weight = svm_weight
        self.feature_weight = feature_weight
        self.C = C
        self.keep_prob = keep_prob
        self.mutate_prob = mutate_prob
        self.iters = iters
        self.feature_extractor_class_name = feature_extractor_class_name
        #TODO 修改映射方法
        if task_type == TaskType.STRUCTURED_DATA_CLASSIFICATION.value:
            self.trainer_class_name = "AKDenseNetForStructruedDataClassificationTrainer"
        elif task_type ==TaskType.STRUCTURED_DATA_REGRESSION.value:
            self.trainer_class_name = "AKDenseNetForStructruedDataRegressionTrainer"
        else:
            raise ValueError(f"The model type '{DenseNetConfig.model_type}' does not support the task type '{task_type}'")
        