from typing import Optional, Union

import numpy as np

class DenseNetConfig():
    model_type = "densenet"
    def __init__(
        self,
        # Hyperparameter tuning config
        # AutoModel参数
        project_name: str = "auto_model",
        max_trials: int = 5,
        directory: Optional[str] = None,
        objective: str = "val_accuracy",
        tuner: str = "greedy",
        overwrite: bool = True,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        # fit()参数配置
        epochs: int = 200,
        validation_split: float = 0.2,
        is_early_stop: Optional[bool] = True,
        # TODO 添加搜索空间参数
        
        # AutoFeatureExtractor config
        feature_num: int = 2, 
        svm_weight: float = 1.0, 
        feature_weight: float = 0, 
        C: Union[float, np.ndarray] = 1.0, 
        keep_prob: float = 0.8, 
        mutate_prob: float = 0.1, 
        iters: int = 1,
        model_class_name = "AKDenseNetForStructredDataClassification",
        **kwargs
    ) -> None:
        # AutoModel参数
        self.project_name = project_name
        self.max_trials = max_trials
        self.directory = directory
        self.objective = objective
        self.tuner = tuner
        self.overwrite = overwrite
        self.seed = seed
        self.max_model_size = max_model_size
        # fit()参数配置
        self.epochs = epochs
        self.validation_split = validation_split
        self.is_early_stop = is_early_stop
        
        # AutoFeatureExtractor config
        self.feature_num = feature_num
        self.svm_weight = svm_weight
        self.feature_weight = feature_weight
        self.C = C
        self.keep_prob = keep_prob
        self.mutate_prob = mutate_prob
        self.iters = iters
        self.model_class_name = model_class_name