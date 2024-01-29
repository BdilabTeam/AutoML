import os
from autotrain import AutoFeatureExtractor, AutoConfig, AutoTrainer

class TestFeatureExtraction:
    def test_dg_for_densenet(self):
        densenet_config = AutoConfig.from_repository(
            trainer_id="structured-data-classification/densenet",
            dp_enable_auto_feature_extract = True,
            tp_epochs=1
        )
        
        trainer = AutoTrainer.from_config(densenet_config)
        
        extractor = AutoFeatureExtractor.from_config(densenet_config)
        output = extractor.extract(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'),
            trainer=trainer, 
        )
        print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")