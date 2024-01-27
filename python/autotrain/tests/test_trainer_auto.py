from autotrain import AutoTrainer

class TestAutoTrainer:
    def test_for_trainer_class(self):
        Trainer = AutoTrainer.for_trainer_class("AKDenseNetForStructruedDataRegressionTrainer")
        assert Trainer is not None
    
    def test_from_repository(self):
        trainer = AutoTrainer.from_repository(
            trainer_id="structured-data-classification/densenet"
        )
        assert trainer is not None