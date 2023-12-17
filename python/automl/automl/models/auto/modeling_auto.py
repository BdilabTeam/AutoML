from collections import OrderedDict
import importlib
from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, model_type_to_module_name


MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("densenet", "AKDenseNetForStructredDataClassification"),
        # ("densenet", "AKDenseNetForStructredDataRegression")
    ]
)

MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING_NAMES)

def _auto_model_class_from_name(class_name: str):
    for model_type, auto_models in MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING_NAMES.items():
        if class_name in auto_models:
            module_name = model_type_to_module_name(model_type)

            module = importlib.import_module(f".{module_name}", "automl.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractor in MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    return None


class AutoModelWithAK():
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
        >>> feature_extractor = AutoModelWithAK.from_class_name("AKDenseNetForStructredDataClassification")
        ```
        """
        return _auto_model_class_from_name(class_name=class_name)
