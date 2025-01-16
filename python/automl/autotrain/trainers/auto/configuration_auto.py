import importlib
from collections import OrderedDict

from ...utils.configuration_utils import BaseTrainerConfig


CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("densenet", "DenseNetTrainerConfig"),
        ("resnet", "ResNetTrainerConfig"),  # 添加这一行
        ("xception", "XceptionTrainerConfig"),
        ("convnet", "ConvNetTrainerConfig"),
        ("yolov8", "YoloV8TrainerConfig")  
    ]
)

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai")
    ]
)

def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

    key = key.replace("-", "_")
    return key

def trainer_id_to_model_type(key: str):
    """Get 'model_type' from a 'task/model' key."""
    key = key.split('/')[1]
    return key

def trainer_id_to_task_type(key: str):
    """Get 'task_type' from a 'task/model' key."""
    key = key.split('/')[0]
    return key

class _LazyConfigMapping(OrderedDict):
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
            raise ValueError(f"'{key}' is already used by a AutoML config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)

class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.for_trainer_class(class_name)` method."
        )
    
    @classmethod
    def from_model_type(cls, model_type: str, **kwargs) -> BaseTrainerConfig:
        """Instantiate one of the configuration classes of the library from model type.
        Examples:
        ```python
        >>> config = AutoConfig.from_model_type("densenet")
        ```
        """
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(**kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )
    
    @classmethod
    def from_repository(cls, trainer_id, **kwargs):
        """Instantiate one of the configuration classes of the library from 'trainer_id' property of the config object.

        Examples:
        ```python
        >>> config = AutoConfig.from_repository("structured-data-classification/densenet")
        ```
        """
        from .trainer_auto import trainer_id_to_trainer_class_name
        model_type = trainer_id_to_model_type(trainer_id)
        kwargs["task_type"] = trainer_id_to_task_type(trainer_id)
        kwargs["trainer_class_name"] = trainer_id_to_trainer_class_name(trainer_id)
        return cls.from_model_type(model_type, **kwargs)
