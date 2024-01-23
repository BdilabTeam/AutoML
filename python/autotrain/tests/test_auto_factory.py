import pytest

from autotrain.models.auto.modeling_auto import MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING
from automl import AutoConfig

class TestAutoFactory:
    def test_get(self):
        for key in MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING.values():
            print(key)
            densenet_config = AutoConfig.from_model_type("densenet")
            print(MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING.get(type(key(densenet_config)), None))
