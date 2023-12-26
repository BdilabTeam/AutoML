import sys
sys.path.append("/Users/treasures/Desktop/AutoML/python/autotrain")
from autotrain.models.auto.modeling_auto import MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING
from automl import AutoConfig


if __name__=="__main__":
    for key in MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING.values():
        print(key)
        densenet_config = AutoConfig.from_model_type("densenet")
        print(MODEL_FOR_STRUCTURED_DATA_CLASSIFICATION_MAPPING.get(type(key(densenet_config)), None))
        