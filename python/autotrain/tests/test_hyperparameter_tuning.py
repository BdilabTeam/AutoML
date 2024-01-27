import os
from autotrain import AutoTrainer, AutoConfig, TaskType, ModelType

class TestHyperparameterTuning:
    
    def test_densenet_for_structured_data_classification_v1(self):
        trainer = AutoTrainer.from_repository(
            trainer_id="structured-data-classification/densenet",
            tp_epochs=1
        )
        output = trainer.train(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv')
        )
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
    
    def test_densenet_for_structured_data_classification_v2(self):
        densenet_config = AutoConfig.from_repository(
            trainer_id="structured-data-classification/densenet",
            tp_epochs=1
        )
        Trainer = AutoTrainer.for_trainer_class(densenet_config.trainer_class_name)
        trainer = Trainer(densenet_config)
        output = trainer.train(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv')
        )
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")

    def test_densenet_for_structured_data_regression(self):
        trainer = AutoTrainer.from_repository(
            trainer_id="structured-data-regression/densenet",
        )
        output = trainer.train(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv')
        )
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
    
    def test_resnet_for_image_classification(self):
        trainer = AutoTrainer.from_repository(
            trainer_id="image-classification/resnet",
        )
        
        output = trainer.train(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'image-classification'), 
        )
        
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")