import importlib
from collections import OrderedDict
from typing import Callable
from .configuration_auto import model_type_to_module_name

TRAIN_FUNC_MAPPING_NAMES = OrderedDict(
    [
        ('densenet', 'train_densenet'),
        ('resnet', 'train_resnet'),
        ('xception', 'train_xception'),
        ('convnet', 'train_convnet'),
        ('yolov8', 'train_yolov8')
    ]
)

class _LazyTrainFuncMapping(OrderedDict):
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
            raise ValueError(f"'{key}' is already used by a AutoML trainer, pick another name.")
        self._extra_content[key] = value

TRAIN_FUNC_MAPPING = _LazyTrainFuncMapping(TRAIN_FUNC_MAPPING_NAMES)

class AutoTrainFunc:
    r"""
    This is a generic train function class that will be got one of the train function of the
    library when created with the [`AutoTrainFunc.from_registry`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTrainFunc is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_registry()` method. or"
            "using the `AutoFeatureExtractor.for_trainer_class()` method."
        )
    
    @classmethod
    def from_model_type(cls, model_type: str) -> Callable:
        """Get one of the train func of the library from model type.
        Examples:
        ```python
        >>> train_resnet_func = AutoTrainFunc.from_model_type("densenet")
        ```
        """
        if model_type in TRAIN_FUNC_MAPPING:
            train_func = TRAIN_FUNC_MAPPING[model_type]
            return train_func
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(TRAIN_FUNC_MAPPING.keys())}"
        )