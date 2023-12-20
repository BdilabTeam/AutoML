from typing import Optional, Union, List

import numpy as np

class DenseNetConfig():
    model_type = "densenet"
    def __init__(
        self,
        task_type: str = "regression",
        # AutoModel config
        project_name: str = "auto_model",
        max_trials: int = 5,
        directory: Optional[str] = None,
        objective: str = "val_loss",
        tuner: str = "greedy",
        overwrite: bool = True,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        # AutoModel.fit() config
        batch_size: int = 32,
        epochs: int = 200,
        validation_split: float = 0.2,
        is_early_stop: Optional[bool] = True,
        # DenseBlock config
        num_layers_search_space: List = [1, 2, 3],
        num_units_search_space: List = [16, 32, 64, 128, 256, 512, 1024],
        use_batchnorm: bool = True,
        dropout_space_search_space: List = [0.0, 0.25, 0.5],
        # ClassificationHead config
        multi_label: bool = False,
        
        # AutoFeatureExtractor config
        use_auto_feature_extract: bool = False,
        feature_num: int = 2, 
        svm_weight: float = 1.0, 
        feature_weight: float = 0, 
        C: Union[float, np.ndarray] = 1.0, 
        keep_prob: float = 0.8, 
        mutate_prob: float = 0.1, 
        iters: int = 1,
        model_class_name = "AKDenseNetForStructruedData",
        **kwargs
    ) -> None:
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
        self.batch_size=batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.is_early_stop = is_early_stop
        # AutoFeatureExtractor
        self.use_auto_feature_extract=use_auto_feature_extract,
        self.feature_num = feature_num
        self.svm_weight = svm_weight
        self.feature_weight = feature_weight
        self.C = C
        self.keep_prob = keep_prob
        self.mutate_prob = mutate_prob
        self.iters = iters
        self.model_class_name = model_class_name
        # DenseBlock
        self.num_layers_search_space = num_layers_search_space
        self.num_units_search_space = num_units_search_space
        self.use_batchnorm = use_batchnorm
        self.dropout_space_search_space = dropout_space_search_space
        # ClassificationHead config
        self.multi_label=multi_label