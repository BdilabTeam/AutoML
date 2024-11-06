import os
import glob
from functools import partial
from typing import Union, Any

import autokeras as ak
from keras_tuner.engine import hyperparameters as hp
from keras.utils import plot_model
import numpy as np
import tensorflow as tf

from .configuration_convnet import ConvNetTrainerConfig
from ...utils import TaskType
from ...utils.trainer_utils import BaseTrainer, Trial, TrialsTracker, BestModelTracker, TrainerTracker, ConfigTracker


class AKBaseTrainerTracker(TrainerTracker):
    pass
    
class AKConvNetMainTrainer:
    def __init__(
        self,
        config: ConvNetTrainerConfig,
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

        convnet_params = {}
        if config.mp_kernel_size:
            convnet_params["kernel_size"] = hp.Choice(name="kernel_size", values=config.mp_kernel_size)
        if config.mp_num_blocks:
            convnet_params["num_blocks"] = hp.Choice(name="num_blocks", values=config.mp_num_blocks)
        if config.mp_num_layers:
            convnet_params["num_layers"] = hp.Choice(name="num_layers", values=config.mp_num_layers)
        if config.mp_filters:
            convnet_params["filters"] = hp.Choice(name="filters", values=config.mp_filters)
        if config.mp_max_pooling:
            convnet_params["max_pooling"] = hp.Boolean(name="max_pooling", default=config.mp_max_pooling)
        if config.mp_separable:
            convnet_params["separable"] = hp.Boolean(name="separable", default=config.mp_separable)
        if config.mp_dropout:
            convnet_params["dropout"] = hp.Choice(name="dropout", values=config.mp_dropout)
        output_node = ak.ConvBlock(**convnet_params)(output_node)
        
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
        auto_model_params["directory"] = config.tp_directory
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
        inputs: str,
        **kwargs
    ) -> AKBaseTrainerTracker:
        data_pipeline_params = {}
        if self._config.dp_batch_size:
            data_pipeline_params["batch_size"] = self._config.dp_batch_size
        if self._config.dp_color_mode:
            data_pipeline_params["color_mode"] = self._config.dp_color_mode
        if self._config.dp_image_size:
            data_pipeline_params["image_size"] = self._config.dp_image_size
        if self._config.dp_interpolation:
            data_pipeline_params["interpolation"] = self._config.dp_interpolation
        if self._config.dp_shuffle:
            data_pipeline_params["shuffle"] = self._config.dp_shuffle
        if self._config.dp_seed:
            data_pipeline_params["seed"] = self._config.dp_seed
        if self._config.dp_validation_split:
            data_pipeline_params["validation_split"] = self._config.dp_validation_split
        train_data = ak.image_dataset_from_directory(
            directory=inputs,
            subset='training',
            **data_pipeline_params
        )
        
        if self._config.task_type == TaskType.IMAGE_CLASSIFICATION.value:
            y_train = np.asarray([label.decode('utf-8') for label in train_data.as_numpy_iterator().next()[1]])
            history = self._auto_fit(train_data)
        elif self._config.task_type == TaskType.IMAGE_REGRESSION.value:
            x_train = train_data.as_numpy_iterator().next()[0]
            y_train = np.asarray([float(label.decode('utf-8')) for label in train_data.as_numpy_iterator().next()[1]])
            history = self._auto_fit(x_train, y_train)
        else:
            raise ValueError(f"`Task type` must be `{TaskType.IMAGE_CLASSIFICATION.value}` or `{TaskType.IMAGE_REGRESSION.value}`")
        
        # Label2Id
        sorted_labels = np.unique(y_train)
        label2ids = {label: index for index, label in enumerate(sorted_labels)}
        id2labels = {index: label for index, label in enumerate(sorted_labels)}
        config_tracker = ConfigTracker(label2ids=label2ids, id2labels=id2labels)
        
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
            trials.append(Trial(
                **trial.get_state(),
                model_graph_path=model_graph_path
            ))
        trials_tracker = TrialsTracker(trials=trials)
        
        return AKBaseTrainerTracker(
            best_model_tracker=best_model_tracker,
            trials_tracker=trials_tracker,
            config_tracker=config_tracker
        )

class AKConvNetForImageClassificationTrainer(BaseTrainer):
    def __init__(self, config: ConvNetTrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.IMAGE_CLASSIFICATION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_CLASSIFICATION.value}'")
        super().__init__(config=config)

        self.trainer = AKConvNetMainTrainer(config=config)
        
    def train(self, inputs: str, *args: Any, **kwds: Any):
        if not self.trainer:
            raise ValueError("No trainer is available")
        trainer_tracker = self.trainer(inputs=inputs)
        self.save_summary(trainer_tracker)

class AKConvNetForImageRegressionTrainer(BaseTrainer):
    def __init__(self, config: ConvNetTrainerConfig, **kwargs):
        if config.task_type != TaskType.IMAGE_REGRESSION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_REGRESSION.value}'")
        super().__init__(config=config)
        
        self.trainer = AKConvNetMainTrainer(config=config)
        
    def train(self, inputs: str, *args: Any, **kwds: Any):
        if not self.trainer:
            raise ValueError("No trainer is available")
        trainer_tracker = self.trainer(inputs=inputs)
        self.save_summary(trainer_tracker)