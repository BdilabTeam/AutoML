import ast
from dataclasses import dataclass, field
from typing import List, Optional, Union

# from ..utils import TrainingArguments, AutoArgumentParser
# from ...models.utils import SUPPORTED_TASK_TYPE, SUPPORTED_MODEL_TYPE
# from ...models.auto import AutoConfig, AutoFeatureExtractor, AutoModelWithAK
from autotrain.trainings.utils import TrainingArguments, AutoArgumentParser
from autotrain.models.utils import SUPPORTED_MODEL_TYPE, SUPPORTED_TASK_TYPE
from autotrain.models.auto import AutoConfig, AutoFeatureExtractor, AutoModelWithAK

@dataclass
class ModelArguments:
    num_layers_search_space: Union[List[int], str] = field(
        default=[1, 2, 3],
        metadata={
            "help": "Layer search space of densenet model"
        }
    )
    num_units_search_space: Union[List[int], str] = field(
        default=[16, 32, 64, 128, 256, 512, 1024],
        metadata={
            "help": "Unit search space of densenet model"
        }
    )
    use_batchnorm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use batch regularization"
        }
    )
    dropout_space_search_space: Union[List[int], str] = field(
        default=[0.0, 0.25, 0.5],
        metadata={
            "help": "Unit search space of densenet model"
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
        
    config = AutoConfig.from_model_type(model_type=train_args.model_type)
    config.task_type = train_args.task_type
    
    # AutoModel config
    if train_args.output_dir:
        config.directory = train_args.output_dir
    if train_args.overwrite:
        config.overwrite = train_args.overwrite
    if train_args.project_name:
        config.project_name = train_args.project_name
    if train_args.max_trials:
        config.max_trials = train_args.max_trials
    if train_args.objective:
        config.objective = train_args.objective
    if train_args.tuner:
        config.tuner = train_args.tuner
    if train_args.batch_size:
        config.batch_size = train_args.batch_size
    if train_args.epochs:
        config.epochs = train_args.epochs
    if train_args.validation_split:
        config.validation_split = train_args.validation_split
    if train_args.is_early_stop:
        config.is_early_stop = train_args.is_early_stop
    
    # Hyperparameters
    if model_args.num_layers_search_space:
        if isinstance(model_args.num_layers_search_space, str):
            model_args.num_layers_search_space = ast.literal_eval(model_args.num_layers_search_space)   
        if not isinstance(model_args.num_layers_search_space, list):
            raise ValueError("The 'num_layers_search_space' must be a list")
        config.num_layers_search_space = model_args.num_layers_search_space
    if model_args.num_units_search_space:
        if isinstance(model_args.num_units_search_space, str):
            model_args.num_units_search_space = ast.literal_eval(model_args.num_units_search_space)   
        if not isinstance(model_args.num_units_search_space, list):
            raise ValueError("The 'num_units_search_space' must be a list")
        config.num_units_search_space = model_args.num_units_search_space
    if model_args.use_batchnorm:
        config.use_batchnorm = model_args.use_batchnorm
    if model_args.dropout_space_search_space:
        if isinstance(model_args.dropout_space_search_space, str):
            model_args.dropout_space_search_space = ast.literal_eval(model_args.dropout_space_search_space)   
        if not isinstance(model_args.dropout_space_search_space, list):
            raise ValueError("The 'dropout_space_search_space' must be a list")
        config.dropout_space_search_space = model_args.dropout_space_search_space
    
    Trainer = AutoModelWithAK.from_class_name(config.model_class_name)
    trainer = Trainer(config=config)
    
    if train_args.do_auto_feature_extract:
        # feature_extractor config
        if feature_extraction_args.iters:
            config.iters = feature_extraction_args.iters
        # TODO 用户可以设置哪些参数？
        FeatureExtractor = AutoFeatureExtractor.from_class_name(config.feature_extractor_class_name)
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
        print(f"{'*'*15}_Train Results Summary:\n{output.results_summary}")
        print(f"{'*'*15}_Model Summary:\n{output.model_summary}")
    
        
        

if __name__ == "__main__":
    main()