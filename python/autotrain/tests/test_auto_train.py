import os
from autotrain import AutoTrainer, AutoConfig

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

class TestAutoTrain:
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
    
    def test_densenet_for_structured_data_classification_v1(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="structured-data-classification",
            trainer_id="structured-data-classification/densenet",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=1,
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'structured-data-classification.csv')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
    
    def test_densenet_for_structured_data_classification_v2(self):
        densenet_config = AutoConfig.from_repository(
            trainer_id="structured-data-classification/densenet",
            tp_epochs=1,
            tp_directory=os.path.dirname(__name__)
        )
        Trainer = AutoTrainer.for_trainer_class(densenet_config.trainer_class_name)
        trainer = Trainer(densenet_config)
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'structured-data-classification.csv')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")

    def test_densenet_for_structured_data_regression(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="structured-data-regression",
            trainer_id="structured-data-regression/densenet",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=1
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'structured-data-regression.csv')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
    
    def test_resnet_for_image_classification(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="image-classification",
            trainer_id="image-classification/resnet",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=2
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-classification')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
    
    def test_resnet_for_image_regression(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="resnet-image-regression",
            trainer_id="image-regression/resnet",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=2,
            tp_overwrite=True,
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-regression')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
        
    def test_xception_for_image_classification(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="xception-image-classification",
            trainer_id="image-classification/xception",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=2
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-classification')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
    
    def test_xception_for_image_regression(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="xception-image-regression",
            trainer_id="image-regression/xception",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=2
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-regression')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
        
    def test_convnet_for_image_classification(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="convnet-image-classification",
            trainer_id="image-classification/convnet",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=2
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-classification')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")
    
    def test_convnet_for_image_regression(self):
        trainer = AutoTrainer.from_repository(
            tp_project_name="convnet-image-regression",
            trainer_id="image-regression/convnet",
            tp_directory=os.path.dirname(__name__),
            tp_epochs=2
        )
        trainer.train(
            inputs=os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-regression')
        )
        summary = trainer.get_summary()
        print(f"{'*'*5}summary:\n{summary}")