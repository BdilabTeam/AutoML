import os
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

from autotrain import AutoTrainFunc

class TestTrainFunc:
    def test_train_densenet(self):
        training_params = {
            'inputs': os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'structured-data-classification.csv'),
            'tp_project_name': 'test',
            'task_type': 'structured-data-classification',
            'model_type': 'densenet',
            'tp_directory': os.path.dirname(__file__),
            "tp_max_trials": 1,
            "tp_tuner": "greedy",
            "tp_batch_size": 32,
            "tp_epochs": 10,
            "tp_validation_split": 0.2
        }
        
        train_func = AutoTrainFunc.from_model_type('densenet')
        train_func(training_params)
    
    def test_train_densenet_with_feature_extract(self):
        training_params = {
            'inputs': os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'structured-data-classification.csv'),
            'tp_project_name': 'test',
            'task_type': 'structured-data-classification',
            'model_type': 'densenet',
            'tp_directory': os.path.dirname(__file__),
            "tp_max_trials": 1,
            "tp_tuner": "greedy",
            "tp_batch_size": 32,
            "tp_epochs": 10,
            "tp_validation_split": 0.2,
            "dp_enable_auto_feature_extract": True,
        }
        
        train_func = AutoTrainFunc.from_model_type('densenet')
        train_func(training_params)
        
        