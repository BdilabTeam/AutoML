from dataclasses import dataclass
from typing import Any, Union, Optional, Dict
from functools import partial

import pandas as pd
import numpy as np

import autokeras as ak
import tensorflow as tf

from .configuration_densenet import DenseNetConfig


@dataclass 
class AKDenseNetForStructredDataOutput():
    metrics: Dict[str, Any] = None
    search_space_summary = None
    results_summary = None
    best_hyperparameters = None
    model_summary = None
    
    
class DenseNetModel():
    def __init__(self, config: DenseNetConfig) -> None:
        pass
    def __call__(
        self, 
        *args: Any, 
        **kwds: Any) -> Any:
        pass


class AKDenseNetMainAutoModel():
    def __init__(
        self,
        config: DenseNetConfig,
        **kwargs
    ):
        # TODO 通过config获取实例化AutoModel的参数、获取fit所需参数
        input_node = ak.StructuredDataInput()
        # output_node - ak.StructuredDataBlock()(input_node)
        output_node = ak.CategoricalToNumerical()(input_node)
        output_node = ak.DenseBlock(use_batchnorm=True)(output_node)
        output_node = ak.ClassificationHead(multi_label=True)(output_node)

        self.auto_model = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            overwrite=config.overwrite, 
            max_trials=config.max_trials
        )
        
        cbs = []
        if config.is_early_stop:
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=100))
        
        self.auto_fit = partial(
            self.auto_model.fit,
            epochs=config.epochs, 
            callbacks=cbs, 
            validation_split=config.validation_split
        )
        
    def __call__(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, str],
        output_metrics: Optional[bool] = None,
        output_best_hyperparameters: Optional[bool] = None,
        output_search_space_summary: Optional[bool] = None,
        output_results_summary: Optional[bool] = None,
        output_model_summary: Optional[bool] = None,
        **kwargs
    ):
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
            raise ValueError("You have to specify `input_ids`")

        # 训练（超参数调优+模型结构搜索）
        self.auto_fit(x=x_train, y=y_train)
        
        # 输出
        output = AKDenseNetForStructredDataOutput()
        if output_metrics:  # 指标
            output.metrics = {}
            metric_keys = self.auto_model.tuner.oracle.get_best_trials(1)[0].metrics.metrics.keys()
            for metric_key in metric_keys:
                metric_value = self.auto_model.tuner.oracle.get_best_trials()[0].metrics.get_statistics(metric_key)["mean"]
                output.metrics.setdefault(metric_key, metric_value)
        if output_best_hyperparameters: # 最优超参数
            output.best_hyperparameters = self.auto_model.tuner.get_best_hyperparameters()[0].values
        if output_search_space_summary: # 搜索空间
            output.search_space_summary = self.auto_model.tuner.oracle.get_space().get_config()
        # TODO 以下调用只会在控制台输出结果，不能将结果保存至‘AKDenseNetForStructredDataOutput’中
        if output_results_summary:    # 训练历史
            output.results_summary = self.auto_model.tuner.results_summary()
        if output_model_summary:    # 模型结构描述
            model = self.auto_model.export_model()
            output.model_summary = model.summary()
        return output
    
    
class AKDenseNetForStructredDataClassification():
    def __init__(self, config: DenseNetConfig, *input, **kwargs) -> None:
        self.densenet = AKDenseNetMainAutoModel(config, name="densenet")
    
    def __call__(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, str],
        output_metrics: Optional[bool] = None,
        output_best_hyperparameters: Optional[bool] = None,
        output_search_space_summary: Optional[bool] = None,
        output_results_summary: Optional[bool] = None,
        output_model_summary: Optional[bool] = None,
        **kwargs
    ) -> Any:
        output = self.densenet(
            inputs=inputs,
            output_metrics=output_metrics,
            output_best_hyperparameters=output_best_hyperparameters,
            output_search_space_summary=output_search_space_summary,
            output_results_summary=output_results_summary,
            output_model_summary=output_model_summary
        )
        # TODO 需要返回什么？
        return output