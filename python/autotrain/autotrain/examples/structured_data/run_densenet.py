import os
import sys
from typing import Optional, List
from dataclasses import dataclass

from autotrain.utils.auto_argparser import AutoArgumentParser
from autotrain.trainers.auto import AutoConfig, AutoFeatureExtractor, AutoTrainer
from autotrain.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DensenetTrainerArguments:
    model_type: str = ""
    task_type: str = ""
    trainer_class_name: str = ""
    inputs: str = ""
    # Data pipeline
    # AutoFeatureExtractor
    dp_enable_auto_feature_extract: bool = False
    dp_feature_num: int = 2
    dp_svm_weight: float = 1.0
    dp_feature_weight: float = 0
    dp_C: float = 1.0
    dp_keep_prob: float = 0.8
    dp_mutate_prob: float = 0.1
    dp_iters: int = 1
    # Model pipeline
    mp_enable_categorical_to_numerical: Optional[bool] = True
    # DenseBlock
    mp_num_layers: Optional[List[int]] = None
    mp_num_units: Optional[List[int]] = None
    mp_use_batchnorm: Optional[bool] = True
    mp_dropout: Optional[List[float]] = None
    # ClassificationHead config
    mp_multi_label: bool = False
    # Train pipeline
    # AutoModel
    tp_project_name: str = "auto_model"
    tp_max_trials: int = 1
    tp_objective: str = "val_loss"
    tp_tuner: str = "greedy"
    tp_overwrite: bool = True
    tp_directory: Optional[str] = None
    tp_seed: Optional[int] = None
    tp_max_model_size: Optional[int] = None
    # AutoModel.fit()
    tp_batch_size: int = 32
    tp_validation_split: float = 0.2
    tp_epochs: Optional[int] = None

def main():
    parser = AutoArgumentParser((DensenetTrainerArguments))
    if len(sys.argv) == 3 and sys.argv[1] == "args_dict":
        trainer_args, = parser.parse_dict(args=sys.argv[2])
    else:
        trainer_args, = parser.parse_args_into_dataclasses()

    trainer_id = os.path.join(trainer_args.task_type, trainer_args.model_type)
    config = AutoConfig.from_repository(trainer_id=trainer_id)

    trainer_args_dict = trainer_args.__dict__
    for key, value in trainer_args_dict.items():
        if key in ['model_type', 'task_type', 'trainer_class_name']:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    
    trainer = AutoTrainer.from_config(config=config)

    if config.dp_enable_auto_feature_extract:
        # TODO 用户可以设置哪些参数？
        feature_extractor = AutoFeatureExtractor.from_config(config)
        feature_extract_output = feature_extractor.extract(
            inputs=trainer_args.inputs,
            trainer=trainer, 
        )
        
        logger.info(f"{'-'*5} Feature extraction history {'-'*5}")
        print(f"{'*'*15}_Best Feature Index:\n{feature_extract_output.best_feature_index}")
    
    logger.info(f"{'-'*5} Start training {'-'*5}")
    trainer.train(inputs=trainer_args.inputs)
        
    train_summary = trainer.get_summary()
    logger.info(f"{'-'*5} Train summary {'-'*5}:\n{train_summary}")
    
if __name__ == "__main__":
    main()