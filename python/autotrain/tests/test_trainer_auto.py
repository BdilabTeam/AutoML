import os
from autotrain import AutoTrainer, AutoConfig

class TestAutoTrainer:
    def test_from_class_name(self):
        densenet_config = AutoConfig.from_model_type("densenet")
        Trainer = AutoTrainer.from_class_name(densenet_config.trainer_class_name)
        trainer = Trainer(densenet_config)
        output = trainer(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'), 
            return_summary_dict=True
        )
        assert output.metrics != None
    
    def test_from_model_type(self):
        trainer = AutoTrainer.from_model_type('densenet')
        output = trainer(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'), 
            return_summary_dict=True
        )
        assert output.metrics != None
    
    def test_from_config(self):
        densenet_config = AutoConfig.from_model_type("densenet")
        trainer = AutoTrainer.from_config(config=densenet_config)
        output = trainer(
            inputs=os.path.join(os.pardir, 'autotrain', 'datasets', 'train.csv'), 
            return_summary_dict=True
        )
        assert output.metrics != None