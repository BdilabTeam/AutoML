import importlib
from typing import Type
from collections import OrderedDict

from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, model_type_to_module_name, AutoConfig
from ...utils.trainer_utils import BaseTrainer
from ...utils.configuration_utils import BaseTrainerConfig


MODEL_FOR_TRAINER_MAPPING_NAMES = OrderedDict(
    [
        (
            "densenet", 
            (
                "AKDenseNetForStructruedDataClassificationTrainer",
                "AKDenseNetForStructruedDataRegressionTrainer"
            )
        ),
        
        (
            "resnet", 
            (
                "AKResNetForImageClassificationTrainer",
                "AKResNetForImageRegressionTrainer"
            )
        ),
        (
            "xception",
            (
                "AKXceptionForImageClassificationTrainer",
                "AKXceptionForImageRegressionTrainer"
            )
        ),
        (
            "convnet",
            (
                "AKConvNetForImageClassificationTrainer",
                "AKConvNetForImageRegressionTrainer"
            )
        ),
        (
            "yolov8",
            (
                "YoloV8ForImageClassificationTrainer",
            )
        )
    ]
)

MODEL_FOR_TRAINER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TRAINER_MAPPING_NAMES)

def _auto_model_class_from_name(class_name: str):
    for model_type, auto_models in MODEL_FOR_TRAINER_MAPPING_NAMES.items():
        if class_name in auto_models:
            module_name = model_type_to_module_name(model_type)

            module = importlib.import_module(f".{module_name}", "autotrain.trainers")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractor in MODEL_FOR_TRAINER_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    return None

TRAINER_MAPPING_NAMES = OrderedDict(
    [
        ("structured-data-classification/densenet", "AKDenseNetForStructruedDataClassificationTrainer"),
        ("structured-data-regression/densenet", "AKDenseNetForStructruedDataRegressionTrainer"),
        ("image-classification/resnet", "AKResNetForImageClassificationTrainer"),
        ("image-regression/resnet", "AKResNetForImageRegressionTrainer"),
        ("image-classification/xception", "AKXceptionForImageClassificationTrainer"),
        ("image-regression/xception", "AKXceptionForImageRegressionTrainer"),
        ("image-classification/convnet", "AKConvNetForImageClassificationTrainer"),
        ("image-regression/convnet", "AKConvNetForImageRegressionTrainer"),
        ("image-classification/yolov8", "YoloV8ForImageClassificationTrainer"),
    ]
)

def trainer_id_to_trainer_class_name(trainer_id: str):
    if trainer_id in TRAINER_MAPPING_NAMES:
        return TRAINER_MAPPING_NAMES[trainer_id]
    
    raise ValueError(
        f"Unrecognized trainer identifier: {trainer_id}. Should contain one of {', '.join(TRAINER_MAPPING_NAMES.keys())}"
    )

def trainer_id_to_module_name(key: str):
    """Converts a 'task/model' key to the corresponding module."""
    key = key.split('/')[1].replace("-", "_")
    return key


class _LazyTrainerMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        if '/' in key:
            module_name = trainer_id_to_module_name(key)
        else:
            module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "autotrain.trainers")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a AutoML trainer, pick another name.")
        self._extra_content[key] = value

TRAINER_MAPPING = _LazyTrainerMapping(TRAINER_MAPPING_NAMES)


class AutoTrainer:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_repository(trainer_id)`"
        )
    
    @classmethod
    def for_trainer_class(cls, class_name: str) -> Type[BaseTrainer]:
        """Get one of the Trainer classes of the library from class name.
        Examples:
        ```python
        >>> trainer = AutoTrainer.for_trainer_class("AKResNetForImageClassificationTrainer")
        ```
        """
        return _auto_model_class_from_name(class_name=class_name)

    @classmethod
    def from_config(cls, config: BaseTrainerConfig, **kwargs) -> Type[BaseTrainer]:
        """Instantiate one of the trainer classes of the library from the config object.
        Examples:
        ```python
        >>> config = AutoConfig.from_repository(trainer_id=trainer_id)
        >>> trainer = AutoTrainer.from_config(config)
        ```
        """
        if isinstance(config, BaseTrainerConfig) and config.trainer_class_name:
            trainer_class = cls.for_trainer_class(config.trainer_class_name)
            return trainer_class(config, **kwargs)
        raise ValueError(
            f"Unrecognized config identifier: ({config})."
        )
    
    @classmethod
    def from_repository(cls, trainer_id: str, **kwargs) -> BaseTrainer:
        """Instantiate one of the trainer classes of the library from 'trainer_id' property of the config object.

        The configuration class to instantiate is selected based on the `trainer_id` property of the config object
        Examples:
        ```python
        >>> config = AutoConfig.from_repository("structured-data-classification/densenet")
        ```
        """
        
        config = AutoConfig.from_repository(trainer_id, **kwargs)
        config.trainer_class_name = trainer_id_to_trainer_class_name(trainer_id)

        if trainer_id in TRAINER_MAPPING:
            trainer_class = TRAINER_MAPPING[trainer_id]
            return trainer_class(config, **kwargs)
        raise ValueError(
            f"Unrecognized trainer_id identifier: ({trainer_id}). Should contain one of {', '.join(TRAINER_MAPPING.keys())}"
        )