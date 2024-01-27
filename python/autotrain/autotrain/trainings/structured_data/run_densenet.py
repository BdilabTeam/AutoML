import ast
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union

from autotrain.trainings.utils import TrainingArguments, AutoArgumentParser
from autotrain.trainers.utils import SUPPORTED_MODEL_TYPE, SUPPORTED_TASK_TYPE
from autotrain.trainers.auto import AutoConfig, AutoFeatureExtractor, AutoTrainer

@dataclass
class ModelArguments:
    num_layers: str = field(
        default=None,
        metadata={
            "help": "Layer search space of densenet model. A list represented by a string"
        }
    )
    num_units: str = field(
        default=None,
        metadata={
            "help": "Unit search space of densenet model. A list represented by a string"
        }
    )
    use_batchnorm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use batch regularization"
        }
    )
    dropout: str = field(
        default=None,
        metadata={
            "help": "Unit search space of densenet model. A list represented by a string"
        }
    )
    multi_label: bool = field(
        default=False,
        metadata={
            "help": "Whether it is a multi-category task. Use only in categorization tasks"
        }
    )

@dataclass
class FeatureExtractionArguments():
    feature_num: Optional[int] = field(
        default=2,
        metadata={
            "help": ""
        }
    )
    svm_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": ""
        }
    )
    feature_weight: Optional[float] = field(
        default=0,
        metadata={
            "help": ""
        }
    )
    C: Optional[float] = field(
        default=1.0,
        metadata={
            "help": ""
        }
    )
    keep_prob: Optional[float] = field(
        default=0.8,
        metadata={
            "help": ""
        }
    )
    mutate_prob: Optional[float] = field(
        default=0.1,
        metadata={
            "help": ""
        }
    )
    iters: Optional[int] = field(
        default=5,
        metadata={
            "help": ""
        }
    )

def main():
    parser = AutoArgumentParser((TrainingArguments, ModelArguments, FeatureExtractionArguments))
    if len(sys.argv) == 3 and sys.argv[1] == "args_dict":
        train_args, model_args, feature_extraction_args = parser.parse_dict(args=sys.argv[2])
    else:
        train_args, model_args, feature_extraction_args = parser.parse_args_into_dataclasses()
        
    # TODO Strict inspection
    if train_args.task_type not in SUPPORTED_TASK_TYPE:
        raise ValueError(f"Do not support the {train_args.task_type} task type, you should check out the supported task type")
    if train_args.model_type not in SUPPORTED_MODEL_TYPE:
        raise ValueError(f"Do not support the {train_args.model_type} model type, you should check out the supported model type")
    if not train_args.train_dir:
        raise ValueError("You must set the 'train_dir' field")
    else:
        inputs = train_args.train_dir
        
    config = AutoConfig.for_config(model_type=train_args.model_type)
    config.task_type = train_args.task_type
    
    # AutoModel config
    if train_args.output_dir:
        config.directory = train_args.output_dir
    if train_args.tp_overwrite:
        config.tp_overwrite = train_args.tp_overwrite
    if train_args.tp_project_name:
        config.tp_project_name = train_args.tp_project_name
    if train_args.tp_max_trials:
        config.tp_max_trials = train_args.tp_max_trials
    if train_args.tp_objective:
        config.tp_objective = train_args.tp_objective
    if train_args.tp_tuner:
        config.tp_tuner = train_args.tp_tuner
    if train_args.batch_size:
        config.batch_size = train_args.batch_size
    if train_args.epochs:
        config.epochs = train_args.epochs
    if train_args.validation_split:
        config.validation_split = train_args.validation_split
    if train_args.is_early_stop:
        config.is_early_stop = train_args.is_early_stop
    
    # Hyperparameters
    if model_args.num_layers:
        if isinstance(model_args.num_layers, str):
            model_args.num_layers = ast.literal_eval(model_args.num_layers)   
        if not isinstance(model_args.num_layers, list):
            raise ValueError("The 'num_layers' must be a list")
        config.num_layers = model_args.num_layers
    if model_args.num_units:
        if isinstance(model_args.num_units, str):
            model_args.num_units = ast.literal_eval(model_args.num_units)   
        if not isinstance(model_args.num_units, list):
            raise ValueError("The 'num_units' must be a list")
        config.num_units = model_args.num_units
    if model_args.use_batchnorm:
        config.use_batchnorm = model_args.use_batchnorm
    if model_args.dropout:
        if isinstance(model_args.dropout, str):
            model_args.dropout = ast.literal_eval(model_args.dropout)   
        if not isinstance(model_args.dropout, list):
            raise ValueError("The 'dropout' must be a list")
        config.dropout = model_args.dropout
    
    Trainer = AutoTrainer.for_trainer_class(config.model_class_name)
    trainer = Trainer(config=config)
    
    if train_args.dp_enable_auto_feature_extract:
        # feature_extractor config
        if feature_extraction_args.iters:
            config.iters = feature_extraction_args.iters
        # TODO 用户可以设置哪些参数？
        FeatureExtractor = AutoFeatureExtractor.for_trainer_class(config.feature_extractor_class_name)
        extractor = FeatureExtractor(config=config)
        output = extractor(
            inputs=inputs,
            trainer=trainer, 
            return_summary_dict=True
        )
        print(f"{'*'*15}_Best Feature Index:\n{output.best_feature_index}")
    
    if train_args.do_auto_hyperparameter_tuning:
        output = trainer(
            inputs=inputs, 
            return_summary_dict=True
        )
        print(f"{'*'*15}_Metrics:\n{output.metrics}")
        print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
        print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
    
if __name__ == "__main__":
    main()