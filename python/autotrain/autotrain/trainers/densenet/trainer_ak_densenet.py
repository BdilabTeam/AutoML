from dataclasses import dataclass
from typing import Any, Union, Optional, Dict
from functools import partial

import pandas as pd
import numpy as np

import autokeras as ak
import tensorflow as tf
from keras_tuner.engine import hyperparameters as hp

from dataclasses import dataclass

from .configuration_densenet import DenseNetTrainerConfig
from ...utils import TaskType
from ...utils.trainer_utils import BaseTrainerOutput, BaseTrainer


@dataclass 
class AKStructruedDataClassificationTrainerOutput(BaseTrainerOutput):
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKStructruedDataRegressionTrainerOutput(BaseTrainerOutput):
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKBaseTrainerOutput(BaseTrainerOutput):
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None


class AKDenseNetMainTrainer():
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
            dense_block_params["num_layers"] = hp.Choice("num_layers", values=config.mp_num_layers, default=1)
        if config.mp_num_units:
            dense_block_params["num_units"] = hp.Choice("num_units", values=config.mp_num_units, default=32)
        if config.mp_dropout:
            dense_block_params["dropout"] = hp.Choice("dropout", values=config.mp_dropout, default=0.0)
        if config.mp_use_batchnorm:
            dense_block_params["use_batchnorm"] = hp.Boolean("use_batchnorm")
        if config.dp_enable_auto_feature_extract:
            dense_block_params["num_layers"] = dense_block_params.pop("num_layers", 1)
            dense_block_params["num_units"] = dense_block_params.pop("num_units", 32)
            dense_block_params["dropout"] = dense_block_params.pop("dropout", 0.0)
        output_node = ak.DenseBlock(**dense_block_params)(output_node)
        
        if config.task_type == TaskType.STRUCTURED_DATA_CLASSIFICATION.value:
            output_node = ak.ClassificationHead(
                multi_label=config.mp_multi_label
            )(output_node)
        elif config.task_type == TaskType.STRUCTURED_DATA_REGRESSION.value:
            output_node = ak.RegressionHead()(output_node)
        else:
            raise ValueError("`task_type` must be `structured_data_classification` or `structured_data_regression`")
        
        auto_model_params = {}
        auto_model_params["project_name"] = config.tp_project_name
        auto_model_params["max_trials"] = config.tp_max_trials
        auto_model_params["objective"] = config.tp_objective
        auto_model_params["tuner"] = config.tp_tuner
        auto_model_params["overwrite"] = config.tp_overwrite
        if config.tp_directory:
            auto_model_params["directory"] = config.tp_directory
        if config.tp_seed:
            auto_model_params["seed"] = config.tp_seed
        if config.tp_max_model_size:
            auto_model_params["max_model_size"] = config.tp_max_model_size
        self.auto_model = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            **auto_model_params
        )
        
        auto_fit_params = {}
        auto_fit_params["batch_size"] = config.tp_batch_size
        auto_fit_params["validation_split"] = config.tp_validation_split
        if config.tp_epochs:
            auto_fit_params["epochs"] = config.tp_epochs
        if config.tp_is_early_stop:
            cbs = []
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=100))
            auto_fit_params["callbacks"] = cbs
        self.auto_fit = partial(
            self.auto_model.fit,
            **auto_fit_params
        )
        
        self.config = config
        
    def __call__(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, str],
        **kwargs
    ) -> Union[AKBaseTrainerOutput, None]:
        # 数据准备
        if inputs is not None:
            if isinstance(inputs, str): # csv file path
                x_y = pd.read_csv(inputs)
                _, features_nums = x_y.shape
                x_train = x_y.iloc[:, 0:(features_nums - 1)].to_numpy()
                y_train = x_y.iloc[:, -1].to_numpy()
            elif isinstance(inputs, np.ndarray):
                raise NotImplementedError
            elif isinstance(inputs, pd.DataFrame):
                print(inputs)
                _, features_nums = inputs.shape
                x_train = inputs.iloc[:, 0:(features_nums - 1)].to_numpy()
                y_train = inputs.iloc[:, -1].to_numpy().astype(int)
            else:
                raise ValueError("`inputs` must be np.ndarray, pd.DataFrame, or str")
        else:
            raise ValueError("You have to specify the `inputs` field")

        # 训练（超参数调优+模型结构搜索）
        self.auto_fit(x=x_train, y=y_train)
        
        # 指标
        metrics = {}
        metric_keys = self.auto_model.tuner.oracle.get_best_trials(1)[0].metrics.metrics.keys()
        for metric_key in metric_keys:
            metric_value = self.auto_model.tuner.oracle.get_best_trials()[0].metrics.get_statistics(metric_key)["mean"]
            metrics.setdefault(metric_key, metric_value)
        # 最优超参数
        best_hyperparameters = self.auto_model.tuner.get_best_hyperparameters()[0].values
        # 搜索空间
        search_space_summary = self.auto_model.tuner.oracle.get_space().get_config()
        # TODO 以下调用只会在控制台输出结果，不能将结果保存至‘AKStructruedDataClassificationTrainerOutput’中
        # if output_results_summary:    # 训练历史
        #    output.results_summary = self.auto_model.tp_tuner.results_summary()
        #if output_model_summary:    # 模型结构描述
        #    model = self.auto_model.export_model()
        #    output.model_summary = model.summary()
        return AKBaseTrainerOutput(
            metrics=metrics,
            best_hyperparameters=best_hyperparameters,
            search_space_summary=search_space_summary
        )
    
    
class AKDenseNetForStructruedDataClassificationTrainer(BaseTrainer):
    def __init__(self, config: DenseNetTrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.STRUCTURED_DATA_CLASSIFICATION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.STRUCTURED_DATA_CLASSIFICATION.value}'")
    
        self.trainer = AKDenseNetMainTrainer(config, name="densenet")
        self.config = config
    
    def train(self, inputs: Union[np.ndarray, pd.DataFrame, str], *args: Any, **kwds: Any) -> AKStructruedDataClassificationTrainerOutput:
        outputs = self.trainer(
            inputs=inputs,
        )
        return AKStructruedDataClassificationTrainerOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )

class AKDenseNetForStructruedDataRegressionTrainer(BaseTrainer):
    def __init__(self, config: DenseNetTrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.STRUCTURED_DATA_REGRESSION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.STRUCTURED_DATA_REGRESSION.value}'")
    
        self.trainer = AKDenseNetMainTrainer(config=config)
        self.config = config
        
    def train(self, inputs: Union[np.ndarray, pd.DataFrame, str], *args: Any, **kwds: Any) -> AKStructruedDataRegressionTrainerOutput:
        outputs = self.trainer(
            inputs=inputs,
        )
        return AKStructruedDataRegressionTrainerOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )