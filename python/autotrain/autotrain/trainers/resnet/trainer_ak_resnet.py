import os
import glob
from functools import partial
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

import autokeras as ak
from keras_tuner.engine import hyperparameters as hp
import tensorflow as tf

from .configuration_resnet import ResNetTrainerConfig
from ...utils import TaskType
from ...utils.trainer_utils import BaseTrainerOutput, BaseTrainer


@dataclass 
class AKBaseTrainerOutput(BaseTrainerOutput):
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKImageClassificationTrainerOutput(BaseTrainerOutput):
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKImageRegressionTrainerOutput(BaseTrainerOutput):
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None
    
class AKResNetMainTrainer:
    def __init__(
        self,
        config: ResNetTrainerConfig,
        **kwargs
    ):
        input_node = ak.ImageInput()
        
        if config.mp_enable_normalization:
            output_node = ak.Normalization()(input_node)
        else:
            output_node = input_node
        
        if config.mp_enable_image_augmentation:
            image_argumentation_params = {}
            if config.mp_translation_factor:
                image_argumentation_params["translation_factor"] = hp.Choice("translation_factor", values=config.mp_translation_factor, default=0.2)
            if config.mp_vertical_flip:
                image_argumentation_params["vertical_flip"] = hp.Boolean("vertical_flip")
            if config.mp_horizontal_flip:
                image_argumentation_params["horizontal_flip"] = hp.Boolean("horizontal_flip")
            if config.mp_rotation_factor:
                image_argumentation_params["rotation_factor"] = hp.Choice("rotation_factor", values=config.mp_rotation_factor, default=0.2)
            if config.mp_zoom_factor:
                image_argumentation_params["zoom_factor"] = hp.Choice("zoom_factor", values=config.mp_zoom_factor, default=0.2)
            if config.mp_contrast_factor:
                image_argumentation_params["contrast_factor"] = hp.Choice("contrast_factor", values=config.mp_zoom_factor)
            output_node = ak.ImageAugmentation(**image_argumentation_params)(output_node)

        resnet_params = {}
        if config.mp_version:
            resnet_params["version"] = config.mp_version
        if config.mp_pretrained:
            resnet_params["pretrained"] = config.mp_pretrained
        output_node = ak.ResNetBlock(**resnet_params)(output_node)
        
        if config.task_type == TaskType.IMAGE_CLASSIFICATION.value:
            output_node = ak.ClassificationHead()(output_node)
        elif config.task_type == TaskType.IMAGE_REGRESSION.value:
            output_node = ak.RegressionHead()(output_node)
        else:
            raise ValueError(f"`Task type` must be `{TaskType.IMAGE_CLASSIFICATION.value}` or `{TaskType.IMAGE_REGRESSION.value}`")
        
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
        inputs: str,
        **kwargs
    ) -> Union[AKBaseTrainerOutput, None]:
        data_pipeline_params = {}
        if self.config.dp_batch_size:
            data_pipeline_params["batch_size"] = self.config.dp_batch_size
        if self.config.dp_color_mode:
            data_pipeline_params["color_mode"] = self.config.dp_color_mode
        if self.config.dp_image_size:
            data_pipeline_params["image_size"] = self.config.dp_image_size
        if self.config.dp_interpolation:
            data_pipeline_params["interpolation"] = self.config.dp_interpolation
        if self.config.dp_shuffle:
            data_pipeline_params["shuffle"] = self.config.dp_shuffle
        if self.config.dp_seed:
            data_pipeline_params["seed"] = self.config.dp_seed
        if self.config.dp_validation_split:
            data_pipeline_params["validation_split"] = self.config.dp_validation_split
        train_data = ak.image_dataset_from_directory(
            directory=inputs,
            subset='training',
            **data_pipeline_params
        )

        self.auto_fit(train_data)

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
        # TODO 以下调用只会在控制台输出结果
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

class AKResNetForImageClassificationTrainer(BaseTrainer):
    def __init__(self, config: ResNetTrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.IMAGE_CLASSIFICATION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_CLASSIFICATION.value}'")

        self.trainer = AKResNetMainTrainer(config=config)
        self.config = config
    
    def train(self, inputs: str, *args: Any, **kwds: Any) -> AKImageClassificationTrainerOutput:
        outputs = self.trainer(
            inputs=inputs,
        )
        return AKImageClassificationTrainerOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )

class AKResNetForImageRegressionTrainer(BaseTrainer):
    def __init__(self, config: ResNetTrainerConfig, **kwargs):
        if config.task_type != TaskType.IMAGE_REGRESSION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_REGRESSION.value}'")
        
        self.trainer = AKResNetMainTrainer(config=config)
        self.config = config
    
    def train(self, inputs: str, *args: Any, **kwds: Any) -> AKImageRegressionTrainerOutput:
        outputs = self.trainer(
            inputs=inputs,
        )
        return AKImageRegressionTrainerOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )
    