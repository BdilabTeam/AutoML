import sys
sys.path.append("/Users/treasures/Desktop/AutoML/python/automl")
from automl import AutoFeatureExtractor, AutoConfig

if __name__ == "__main__":
    densenet_config = AutoConfig.from_model_type("densenet")
    Extractor = AutoFeatureExtractor.from_class_name("DenseNetFeatureExtractor")
    extractor = Extractor(densenet_config)
    output = extractor(
        inputs="/Users/treasures/Desktop/AutoML/python/automl/automl/datasets/train.csv",
        output_best_feature_index=True,
    )
    print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")