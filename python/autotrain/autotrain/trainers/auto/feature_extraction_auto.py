import importlib
from collections import OrderedDict

from .auto_factory import _LazyAutoMapping
from .configuration_auto import model_type_to_module_name, CONFIG_MAPPING_NAMES
from ...utils.feature_extraction_utils import BaseFeatureExtractor
from ...utils.configuration_utils import BaseTrainerConfig


FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    [
        ("densenet", "GAForDenseNetFeatureExtractor")
    ]
)

FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)

def _feature_extractor_class_from_name(class_name: str):
    for module_type, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_type)

            module = importlib.import_module(f".{module_name}", "autotrain.trainers")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractor in FEATURE_EXTRACTOR_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    return None


class AutoFeatureExtractor:
    r"""
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_registry`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_registry(model_name_or_path)` method. or"
            "using the `AutoFeatureExtractor.for_trainer_class(class_name)` method."
        )
    
    @classmethod
    def for_feature_extractor_class(cls, class_name: str):
        """Get one of the 'Feature Exreactor' classes of the library from class name.
        Examples:
        ```python
        >>> feature_extractor = AutoFeatureExtractor.for_feature_extractor_class("DenseNetFeatureExtractor")
        ```
        """
        return _feature_extractor_class_from_name(class_name=class_name)
    
    @classmethod
    def from_config(cls, config: BaseTrainerConfig, **kwargs) -> BaseFeatureExtractor:
        """Instantiate one of the feature-extractor classes of the library from the config object.
        Examples:
        ```python
        >>> config = AutoConfig.from_repository(trainer_id=trainer_id)
        >>> feature_extractor = AutoFeatureExtractor.from_config(config)
        ```
        """
        if isinstance(config, BaseTrainerConfig) and config.dp_feature_extractor_class_name:
            feature_extractor_class = cls.for_feature_extractor_class(config.dp_feature_extractor_class_name)
            return feature_extractor_class(config, **kwargs)
        raise ValueError(
            f"Unrecognized config identifier: ({config})."
        )