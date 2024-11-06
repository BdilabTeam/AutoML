import os
from typing import Any
from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from keras_tuner.engine import hyperparameters as hp
from keras.utils import plot_model
from sklearn.model_selection import train_test_split


from .configuration_densenet import DenseNetTrainerConfig
from ...utils import TaskType
from ...utils.trainer_utils import BaseTrainer, BestModelTracker, Trial, TrialsTracker, TrainerTracker, ConfigTracker

class AKBaseTrainerTracker(TrainerTracker):
    pass

class AKDenseNetMainTrainer:
    def __init__(
        self,
        config: DenseNetTrainerConfig,
        **kwargs
    ):
        input_node = ak.StructuredDataInput()
        if config.mp_enable_categorical_to_numerical:
            output_node = ak.CategoricalToNumerical()(input_node)
        else:
            output_node = input_node
        
        dense_block_params = {}
        if config.mp_num_layers:
            dense_block_params["num_layers"] = hp.Choice("num_layers", values=config.mp_num_layers)
        if config.mp_num_units:
            dense_block_params["num_units"] = hp.Choice("num_units", values=config.mp_num_units)
        if config.mp_dropout:
            dense_block_params["dropout"] = hp.Choice("dropout", values=config.mp_dropout)
        if config.mp_use_batchnorm:
            dense_block_params["use_batchnorm"] = hp.Boolean("use_batchnorm")
        if config.dp_enable_auto_feature_extract:
            dense_block_params["num_layers"] = dense_block_params.pop("num_layers").default
            dense_block_params["num_units"] = dense_block_params.pop("num_units").default
            dense_block_params["dropout"] = dense_block_params.pop("dropout").default
        output_node = ak.DenseBlock(**dense_block_params)(output_node)
        
        if config.task_type == TaskType.STRUCTURED_DATA_CLASSIFICATION.value:
            output_node = ak.ClassificationHead(
                multi_label=config.mp_multi_label
            )(output_node)
        elif config.task_type == TaskType.STRUCTURED_DATA_REGRESSION.value:
            output_node = ak.RegressionHead()(output_node)
        else:
            raise ValueError(f"`task_type` must be `{TaskType.STRUCTURED_DATA_CLASSIFICATION.value}` or `{TaskType.STRUCTURED_DATA_REGRESSION.value}`")
        
        auto_model_params = {}
        auto_model_params["project_name"] = config.tp_project_name
        auto_model_params["directory"] = config.tp_directory
        auto_model_params["max_trials"] = config.tp_max_trials
        auto_model_params["tuner"] = config.tp_tuner
        auto_model_params["overwrite"] = config.tp_overwrite
        
        if config.tp_objective:
            auto_model_params["objective"] = config.tp_objective
        if config.tp_seed:
            auto_model_params["seed"] = config.tp_seed
        if config.tp_max_model_size:
            auto_model_params["max_model_size"] = config.tp_max_model_size
        self._auto_model = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            **auto_model_params
        )
        
        auto_fit_params = {}
        auto_fit_params["batch_size"] = config.tp_batch_size
        auto_fit_params["validation_split"] = config.tp_validation_split
        if config.tp_epochs:
            auto_fit_params["epochs"] = config.tp_epochs
        self._auto_fit = partial(
            self._auto_model.fit,
            **auto_fit_params
        )
        
        self._config = config
        
    def __call__(
        self,
        inputs,
        **kwargs
    ) -> AKBaseTrainerTracker:
        # 数据准备
        if inputs is not None:
            if isinstance(inputs, str): # csv file path
                X_y = pd.read_csv(inputs)
                _, features_nums = X_y.shape
                X = X_y.iloc[:, 0:(features_nums - 1)].to_numpy()
                y = X_y.iloc[:, -1].to_numpy()
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            elif isinstance(inputs, pd.DataFrame):
                _, features_nums = inputs.shape
                X = inputs.iloc[:, :(features_nums - 1)].to_numpy()
                y = inputs.iloc[:, -1].to_numpy()
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            else:
                raise ValueError("`inputs` must be pd.DataFrame or str")
            
            sorted_labels = np.unique([str(label) for label in y])
            label2ids = {label: index for index, label in enumerate(sorted_labels)}
            id2labels = {index: label for index, label in enumerate(sorted_labels)}
            config_tracker = ConfigTracker(label2ids=label2ids, id2labels=id2labels)
        else:
            raise ValueError("You have to specify the `inputs` field")

        # 训练（超参数调优+模型结构搜索）
        # history = self._auto_fit(dataset)
        history = self._auto_fit(x=x_train, y=y_train, validation_data=(x_val, y_val),)
        
        best_keras_model = self._auto_model.tuner.get_best_model()
        try:
            model_graph_path = os.path.join(self._auto_model.tuner.best_model_path, 'model.png')
            plot_model(best_keras_model, to_file=model_graph_path, show_layer_activations=True, show_dtype=True, show_shapes=True, show_layer_names=False)
        except:
            model_graph_path = None

        best_model_tracker = BestModelTracker(
            history=history.history,
            hyperparameters=self._auto_model.tuner.get_best_hyperparameters().pop().get_config(),
            model_graph_path=model_graph_path
        )
        
        max_trials = self._config.tp_max_trials
        trials = []
        models = self._auto_model.tuner.get_best_models(max_trials)
        index = 0
        for trial in self._auto_model.tuner.oracle.get_best_trials(max_trials):
            try:
                model_graph_path = os.path.join(self._auto_model.tuner.get_trial_dir(trial_id=trial.trial_id), 'model.png')
                plot_model(model=models[index], to_file=model_graph_path, show_layer_activations=True, show_dtype=True, show_shapes=True, show_layer_names=False)
            except:
                model_graph_path = None
            index += 1
            trials.append(
                Trial(**trial.get_state(), model_graph_path=model_graph_path)
            )
        trials_tracker = TrialsTracker(trials=trials)
        
        return AKBaseTrainerTracker(
            best_model_tracker=best_model_tracker,
            trials_tracker=trials_tracker,
            config_tracker=config_tracker
        )
    
    
class AKDenseNetForStructruedDataClassificationTrainer(BaseTrainer):
    def __init__(self, config: DenseNetTrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.STRUCTURED_DATA_CLASSIFICATION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.STRUCTURED_DATA_CLASSIFICATION.value}'")
        super().__init__(config=config)
    
        self.trainer = AKDenseNetMainTrainer(config)
        
    def train(self, inputs, *args: Any, **kwds: Any):
        if not self.trainer:
            raise ValueError("No trainer is available")
        trainer_tracker = self.trainer(inputs=inputs)
        self.save_summary(trainer_tracker)
        

class AKDenseNetForStructruedDataRegressionTrainer(BaseTrainer):
    def __init__(self, config: DenseNetTrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.STRUCTURED_DATA_REGRESSION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.STRUCTURED_DATA_REGRESSION.value}'")
        super().__init__(config=config)
    
        self.trainer = AKDenseNetMainTrainer(config=config)
        
    def train(self, inputs, *args: Any, **kwds: Any):
        if not self.trainer:
            raise ValueError("No trainer is available")
        trainer_tracker = self.trainer(inputs=inputs)
        self.save_summary(trainer_tracker)