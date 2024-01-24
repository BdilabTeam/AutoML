import os
from autotrain import AutoFeatureExtractor, AutoConfig, AutoTrainer

class TestFeatureExtract:
    def test_dg_with_densenet(self):
        densenet_config = AutoConfig.from_model_type("densenet")
        densenet_config.use_auto_feature_extract = True
        
        Trainer = AutoTrainer.from_class_name(densenet_config.trainer_class_name)
        trainer = Trainer(densenet_config)
        
        Extractor = AutoFeatureExtractor.from_class_name("DenseNetFeatureExtractor")
        extractor = Extractor(config=densenet_config)
        output = extractor(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'),
            trainer=trainer, 
            return_summary_dict=True
        )
        print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")