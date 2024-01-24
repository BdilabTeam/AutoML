from typing import Callable
from collections import OrderedDict
import importlib
from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, model_type_to_module_name, AutoConfig

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
    ]
)

MODEL_FOR_TRAINER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TRAINER_MAPPING_NAMES)

def _auto_model_class_from_name(class_name: str):
    for model_type, auto_models in MODEL_FOR_TRAINER_MAPPING_NAMES.items():
        print(auto_models)
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


class AutoTrainer():
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_class_name(class_name)`"
        )
    
    @classmethod
    def from_class_name(cls, class_name):
        """
        Examples:
        ```python
        >>> trainer = AutoTrainer.from_class_name("AKResNetForImageClassificationTrainer")
        ```
        """
        return _auto_model_class_from_name(class_name=class_name)
    
    @classmethod
    def from_model_type(cls, model_type) -> Callable:
        """
        Examples:
        ```python
        >>> trainer = AutoTrainer.rom_model_type("densenet")
        ```
        """
        config = AutoConfig.from_model_type(model_type=model_type)
        Trainer = cls.from_class_name(config.trainer_class_name)
        return Trainer(config)
    
    @classmethod
    def from_config(cls, config) -> Callable:
        """
        Examples:
        ```python
        >>> trainer = AutoTrainer.from_config(Config)
        ```
        """
        Trainer = cls.from_class_name(config.trainer_class_name)
        return Trainer(config)
