import os
from autotrain import AutoTrainer, AutoConfig

class TestHyperparameterTuning:
    
    def test_densenet(self):
        densenet_config = AutoConfig.from_model_type("densenet")
        
        trainer = AutoTrainer.from_config(densenet_config)
        
        output = trainer(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'), 
            return_summary_dict=True
        )
        
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
        print(f"{'*'*15}_Train Results Summary:\n{output.results_summary}")
        print(f"{'*'*15}_Model Summary:\n{output.model_summary}")
    
    def test_resnet(self):
        resnet_config = AutoConfig.from_model_type("resnet")
        resnet_config.epochs = 2
        
        trainer = AutoTrainer.from_config(config=resnet_config)
        
        output = trainer(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'image-classification'), 
            return_summary_dict=True
        )
        
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
        print(f"{'*'*15}_Train Results Summary:\n{output.results_summary}")
        print(f"{'*'*15}_Model Summary:\n{output.model_summary}")