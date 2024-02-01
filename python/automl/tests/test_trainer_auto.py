from autotrain import AutoTrainer, AutoConfig

class TestAutoTrainer:
    def test_for_trainer_class(self):
        trainer_class = AutoTrainer.for_trainer_class("AKDenseNetForStructruedDataRegressionTrainer")
        assert trainer_class is not None
    
    def test_from_config(self):
        config = AutoConfig.from_repository(trainer_id="structured-data-classification/densenet")
        trainer = AutoTrainer.from_config(config)
        assert trainer is not None
    
    def test_from_repository(self):
        trainer = AutoTrainer.from_repository(
            trainer_id="structured-data-classification/densenet"
        )
        assert trainer is not None