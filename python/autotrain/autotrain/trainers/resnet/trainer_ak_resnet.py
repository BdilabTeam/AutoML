import os
import glob
from functools import partial
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

import autokeras as ak
from keras_tuner.engine import hyperparameters as hp
import tensorflow as tf

from .configuration_resnet import ResNetConfig
from ..utils import TaskType


@dataclass 
class AKImageClassificationTrainerOutput():
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKImageRegressionTrainerOutput():
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None

@dataclass 
class AKBaseTrainerOutput():
    metrics: Dict[str, Any] = None
    search_space_summary: Dict[str, Any] = None
    best_hyperparameters: Dict[str, Any] = None
    results_summary: Any = None
    model_summary: Any = None
    
class AKResNetMainTrainer():
    def __init__(
        self,
        config: ResNetConfig,
        **kwargs
    ):
        input_node = ak.ImageInput()
        
        if config.enable_normalization:
            output_node = ak.Normalization()(input_node)
        else:
            output_node = input_node
        
        if config.enable_image_augmentation:
            image_argumentation_params = {}
            if config.translation_factor:
                image_argumentation_params["translation_factor"] = hp.Choice("translation_factor", values=config.translation_factor, default=0.2)
            if config.vertical_flip:
                image_argumentation_params["vertical_flip"] = hp.Boolean("vertical_flip")
            if config.horizontal_flip:
                image_argumentation_params["horizontal_flip"] = hp.Boolean("horizontal_flip")
            if config.rotation_factor:
                image_argumentation_params["rotation_factor"] = hp.Choice("rotation_factor", values=config.rotation_factor, default=0.2)
            if config.zoom_factor:
                image_argumentation_params["zoom_factor"] = hp.Choice("zoom_factor", values=config.zoom_factor, default=0.2)
            if config.contrast_factor:
                image_argumentation_params["contrast_factor"] = hp.Choice("contrast_factor", values=config.zoom_factor)
            output_node = ak.ImageAugmentation(**image_argumentation_params)(output_node)

        resnet_params = {}
        if config.version:
            resnet_params["version"] = config.version
        if config.pretrained:
            resnet_params["pretrained"] = config.pretrained
        output_node = ak.ResNetBlock(**resnet_params)(output_node)
        
        if config.task_type == TaskType.IMAGE_CLASSIFICATION.value:
            output_node = ak.ClassificationHead()(output_node)
        elif config.task_type == TaskType.IMAGE_REGRESSION.value:
            output_node = ak.RegressionHead()(output_node)
        else:
            raise ValueError("`task_type` must be `structured_data_classification` or `structured_data_regression`")
        
        auto_model_params = {}
        auto_model_params["project_name"] = config.project_name
        auto_model_params["max_trials"] = config.max_trials
        auto_model_params["objective"] = config.objective
        auto_model_params["tuner"] = config.tuner
        auto_model_params["overwrite"] = config.overwrite
        if config.directory:
            auto_model_params["directory"] = config.directory
        if config.seed:
            auto_model_params["seed"] = config.seed
        if config.max_model_size:
            auto_model_params["max_model_size"] = config.max_model_size
        self.auto_model = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            **auto_model_params
        )
        
        auto_fit_params = {}
        auto_fit_params["batch_size"] = config.batch_size
        auto_fit_params["validation_split"] = config.validation_split
        if config.epochs:
            auto_fit_params["epochs"] = config.epochs
        if config.is_early_stop:
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
        return_summary_dict: Optional[bool],
        **kwargs
    ):
        # 数据准备
        train_dir = inputs
        items = os.listdir(train_dir)
        # 获取'文件夹'名称
        folder_names = [item for item in items if os.path.isdir(os.path.join(train_dir, item))]

        file_paths = []
        labels = []
        for folder_name in folder_names:
            files = glob.glob(os.path.join(train_dir, folder_name, '*'))
            file_paths.extend(files)
            labels.extend([folder_name] * len(files))
            
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        def load_image(file_path, label):
            image = tf.io.read_file(file_path)
            image = tf.image.decode_image(image, channels=3)
            return image, label

        dataset = dataset.map(load_image)

        x_train = []
        y_train = []
        for x, y in dataset:
            x_train.append(x)
            y_train.append(y)
        # 将特征和标签转换为张量
        x_train = np.asarray(tf.stack(x_train))
        y_train = np.asarray(tf.stack(y_train))
        
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
        # TODO 以下调用只会在控制台输出结果，不能将结果保存至‘AKStructruedDataClassificationTrainerOutput’中
        # if output_results_summary:    # 训练历史
        #    output.results_summary = self.auto_model.tuner.results_summary()
        #if output_model_summary:    # 模型结构描述
        #    model = self.auto_model.export_model()
        #    output.model_summary = model.summary()
        return AKBaseTrainerOutput(
            metrics=metrics,
            best_hyperparameters=best_hyperparameters,
            search_space_summary=search_space_summary
        )

class AKResNetForImageClassificationTrainer():
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        if config.task_type != TaskType.IMAGE_CLASSIFICATION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_CLASSIFICATION.value}'")
        
        self.trainer = AKResNetMainTrainer(config, name="resnet")
        self.config = config
    
    def __call__(
        self,
        inputs: str,
        return_summary_dict: Optional[bool] = None,
        **kwargs
    ) -> Any:
        outputs = self.trainer(
            inputs=inputs,
            return_summary_dict=return_summary_dict,
        )
        if not return_summary_dict:
            return None
        return AKImageClassificationTrainerOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )

class AKResNetForImageRegressionTrainer():
    def __init__(self, config: ResNetConfig, **kwargs):
        if config.task_type != TaskType.IMAGE_REGRESSION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_REGRESSION.value}'")
        
        self.trainer = AKResNetMainTrainer(config, name="resnet")
        self.config = config
    
    def __call__(
        self,
        inputs: str,
        return_summary_dict: Optional[bool] = None,
        **kwargs
    ) -> Any:
        outputs = self.trainer(
            inputs=inputs,
            return_summary_dict=return_summary_dict,
        )
        if not return_summary_dict:
            return None
        return AKImageClassificationTrainerOutput(
            metrics=outputs.metrics,
            best_hyperparameters=outputs.best_hyperparameters,
            search_space_summary=outputs.search_space_summary,
            results_summary=outputs.results_summary,
            model_summary=outputs.model_summary,
        )