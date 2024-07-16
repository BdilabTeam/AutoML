import os
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

from autotrain import AutoTrainFunc

MINIO_ENDPOINT = "124.70.188.119:32090"
MINIO_ACCESS_KEY= "lUAFGmD6TL57zCcFJTmo"
MINIO_SECRET_KEY = "ZMIG5DToDNtL5I86oeEDcvPhIE5PhlFe67oMVN0a"

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
            "tp_validation_split": 0.2,
            "experiment_name": "densenet",
            "minio_config": {
                "minio_endpoint": MINIO_ENDPOINT,
                "minio_access_key": MINIO_ACCESS_KEY,
                "minio_secret_key": MINIO_SECRET_KEY
            }
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
            "experiment_name": "densenet",
            "minio_config": {
                "minio_endpoint": MINIO_ENDPOINT,
                "minio_access_key": MINIO_ACCESS_KEY,
                "minio_secret_key": MINIO_SECRET_KEY
            }
        }
        
        train_func = AutoTrainFunc.from_model_type('densenet')
        train_func(training_params)
        
    def test_train_resnet_for_image_classification(self):
        training_params = {
            'inputs': os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-classification'),
            'tp_project_name': 'resnet-image-classification',
            'task_type': 'image-classification',
            'model_type': 'resnet',
            'tp_directory': os.path.dirname(__file__),
            "tp_max_trials": 1,
            "tp_tuner": "greedy",
            "tp_batch_size": 32,
            "tp_epochs": 10,
            "tp_validation_split": 0.2,
            "experiment_name": "resnet",
            "minio_config": {
                "minio_endpoint": MINIO_ENDPOINT,
                "minio_access_key": MINIO_ACCESS_KEY,
                "minio_secret_key": MINIO_SECRET_KEY
            }
        }
        
        train_func = AutoTrainFunc.from_model_type('resnet')
        train_func(training_params)
        
    def test_train_convnet_for_image_classification(self):
        training_params = {
            'inputs': os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-classification'),
            'tp_project_name': 'convnet-image-classification',
            'task_type': 'image-classification',
            'model_type': 'convnet',
            'tp_directory': os.path.dirname(__file__),
            "tp_max_trials": 1,
            "tp_tuner": "greedy",
            "tp_batch_size": 32,
            "tp_epochs": 10,
            "tp_validation_split": 0.2,
            "experiment_name": "convnet",
            "minio_config": {
                "minio_endpoint": MINIO_ENDPOINT,
                "minio_access_key": MINIO_ACCESS_KEY,
                "minio_secret_key": MINIO_SECRET_KEY
            }
        }
        
        train_func = AutoTrainFunc.from_model_type('convnet')
        train_func(training_params) 
    
    def test_train_xception_for_image_classification(self):
        training_params = {
            'inputs': os.path.join(PARENT_DIR, 'autotrain', 'datasets', 'image-classification'),
            'tp_project_name': 'xception-image-classification',
            'task_type': 'image-classification',
            'model_type': 'xception',
            'tp_directory': os.path.dirname(__file__),
            "tp_max_trials": 1,
            "tp_tuner": "greedy",
            "tp_batch_size": 32,
            "tp_epochs": 10,
            "tp_validation_split": 0.2,
            "experiment_name": "xception",
            "minio_config": {
                "minio_endpoint": MINIO_ENDPOINT,
                "minio_access_key": MINIO_ACCESS_KEY,
                "minio_secret_key": MINIO_SECRET_KEY
            }
        }
        
        train_func = AutoTrainFunc.from_model_type('xception')
        train_func(training_params) 