import copy
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from .hub import(
    download_url,
    is_offline_mode,
    is_remote_url,
)

from .logging import get_logger

CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
logger = get_logger(__name__)

class FeatureExtractionMixin():
    """
    This is a feature extraction mixin used to provide saving/loading functionality for 'structured data' feature
    extractors.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
    
    @staticmethod
    def from_registry(cls, model_name_or_path: Union[str, os.PathLike]):
        r"""
        Instantiate a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a feature extractor, *e.g.* a
        derived class of [`StructuredFeatureFeatureExtractor`].
        """
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)
    
    @classmethod
    def get_feature_extractor_dict(
        cls, 
        model_name_or_path: Union[str, os.PathLike], 
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        local_files_only = kwargs.pop("local_files_only", False)

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        model_name_or_path = str(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)
        if os.path.isdir(model_name_or_path):
            feature_extractor_file = os.path.join(model_name_or_path, FEATURE_EXTRACTOR_NAME)
        if os.path.isfile(model_name_or_path):
            resolved_feature_extractor_file = model_name_or_path
            is_local = True
        elif is_remote_url(model_name_or_path):
            feature_extractor_file = model_name_or_path
            resolved_feature_extractor_file = download_url(model_name_or_path)
        else:
            feature_extractor_file = FEATURE_EXTRACTOR_NAME

        try:
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)
        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_feature_extractor_file}")
        else:
            logger.info(
                f"loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs):
        """Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)

        logger.info(f"Feature extractor {feature_extractor}")
 
        return feature_extractor
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary. 
        
            Returns:
                `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        return output