import sys
sys.path.append("/Users/treasures/AllProjects/Projects/Git/Bdilab/AutoML/python/autotrain")
from autotrain import AutoFeatureExtractor, AutoConfig, AutoModelWithAK

if __name__ == "__main__":
    densenet_config = AutoConfig.from_model_type("densenet")
    densenet_config.use_auto_feature_extract = True
    
    Trainer = AutoModelWithAK.from_class_name(densenet_config.model_class_name)
    trainer = Trainer(densenet_config)
    
    Extractor = AutoFeatureExtractor.from_class_name("DenseNetFeatureExtractor")
    extractor = Extractor(config=densenet_config)
    output = extractor(
        inputs="/Users/treasures/AllProjects/Projects/Git/Bdilab/AutoML/python/automl/automl/datasets/train.csv",
        trainer=trainer, 
        return_summary_dict=True
    )
    print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")