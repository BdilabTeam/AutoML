import os
from autotrain import AutoFeatureExtractor, AutoConfig, AutoTrainer

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

class TestFeatureExtraction:
    def test_dg_for_densenet(self):
        densenet_config = AutoConfig.from_repository(
            tp_project_name="structured-data-classification",
            trainer_id="structured-data-classification/densenet",
            dp_enable_auto_feature_extract=True,
            tp_epochs=1,
            tp_directory=os.path.dirname(__file__)
        )
        
        trainer = AutoTrainer.from_config(densenet_config)
        
        extractor = AutoFeatureExtractor.from_config(densenet_config)
        output = extractor.extract(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'structured-data-classification.csv'),
            trainer=trainer, 
        )
        print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")