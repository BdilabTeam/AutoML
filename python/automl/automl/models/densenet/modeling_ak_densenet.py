from dataclasses import dataclass
from typing import Any, Union, Optional, Dict
from functools import partial

import pandas as pd
import numpy as np

import autokeras as ak
import tensorflow as tf
from keras_tuner.engine import hyperparameters as hp

from dataclasses import dataclass

from .configuration_densenet import DenseNetConfig


@dataclass 
class AKStructruedDataModelOutput():
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKBaseModelOutput():
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None
    
class AKDenseNetMainAutoModel():
    def __init__(
        self,
        config: DenseNetConfig,
        **kwargs
    ):
        num_layers_search_space = hp.Choice("num_layers", values=config.num_layers_search_space, default=2)
        num_units_search_space = hp.Choice("num_units", values=config.num_units_search_space, default=32)
        dropout_space_search_space = hp.Choice("dropout", values=config.dropout_space_search_space, default=0.0)
        if config.use_auto_feature_extract:
            num_layers_search_space = num_layers_search_space.default
            num_units_search_space = num_units_search_space.default
            dropout_space_search_space = dropout_space_search_space.default

        structured_data_input = ak.StructuredDataInput()
        structured_data_output = ak.CategoricalToNumerical()(structured_data_input)
        structured_data_output = ak.DenseBlock(
            num_layers=num_layers_search_space,
            num_units=num_units_search_space,
            use_batchnorm=config.use_batchnorm,
            dropout=dropout_space_search_space,
        )(structured_data_input)
        classification_output = ak.ClassificationHead(
            multi_label=config.multi_label
        )(structured_data_output)
        # regression_output = ak.RegressionHead()(structured_data_output)

        self.auto_model = ak.AutoModel(
            inputs=structured_data_input, 
            # outputs=[classification_output, regression_output], 
            outputs=[classification_output], 
            overwrite=config.overwrite, 
            max_trials=config.max_trials
        )
        
        cbs = []
        if config.is_early_stop:
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=100))
        
        self.auto_fit = partial(
            self.auto_model.fit,
            batch_size=config.batch_size,
            epochs=config.epochs, 
            callbacks=cbs, 
            validation_split=config.validation_split
        )
        
        self.config = config
        
    def __call__(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, str],
        return_summary_dict: Optional[bool],
        **kwargs
    ):
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
        
        if not return_summary_dict:
            return None
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
        # TODO 以下调用只会在控制台输出结果，不能将结果保存至‘AKStructruedDataModelOutput’中
        # if output_results_summary:    # 训练历史
        #    output.results_summary = self.auto_model.tuner.results_summary()
        #if output_model_summary:    # 模型结构描述
        #    model = self.auto_model.export_model()
        #    output.model_summary = model.summary()
        return AKBaseModelOutput(
            metrics=metrics,
            best_hyperparameters=best_hyperparameters,
            search_space_summary=search_space_summary
        )
    
    
class AKDenseNetForStructruedData():
    def __init__(self, config: DenseNetConfig, **kwargs) -> None:
        self.densenet = AKDenseNetMainAutoModel(config, name="densenet")
        self.config = config
    
    def __call__(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, str],
        return_summary_dict: Optional[bool] = None,
        **kwargs
    ) -> Any:
        outputs = self.densenet(
            inputs=inputs,
            return_summary_dict=return_summary_dict,
        )
        if not return_summary_dict:
            return None
        return AKStructruedDataModelOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )