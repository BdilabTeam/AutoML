import os
from autotrain import AutoFeatureExtractor, AutoConfig, AutoTrainer

class TestFeatureExtract:
    def test_dg_with_densenet(self):
        densenet_config = AutoConfig.from_repository(
            trainer_id="structured-data-classification/densenet",
            dp_enable_auto_feature_extract = True,
            tp_epochs=1
        )
        
        Trainer = AutoTrainer.for_trainer_class(densenet_config.trainer_class_name)
        trainer = Trainer(densenet_config)
        
        Extractor = AutoFeatureExtractor.for_feature_extractor_class(densenet_config.dp_feature_extractor_class_name)
        extractor = Extractor(config=densenet_config)
        output = extractor(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'),
            trainer=trainer, 
        )
        print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")