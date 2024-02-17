import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from autotrain.utils.auto_argparser import AutoArgumentParser
from autotrain.trainers.auto import AutoConfig, AutoTrainer
from autotrain.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ResNetTrainerArguments:
    model_type:str = field(default='resnet')
    task_type: str = ""
    trainer_class_name: str = ""
    inputs: str = ""
    # Data Pipeline
    dp_batch_size: Optional[int] = None
    dp_color_mode: Optional[str] = None
    dp_image_size: Optional[Tuple[float, float]] = None
    dp_interpolation: Optional[str] = None
    dp_shuffle: Optional[bool] = None
    dp_seed: Optional[int] = None
    dp_validation_split: Optional[float] = None
    dp_subset: Optional[str] = None
    # Model Pipeline
    # Normalization
    mp_enable_normalization: bool = field(default=True)
    # ImageAugmentation
    mp_enable_image_augmentation: bool = field(default=True)
    mp_translation_factor: Optional[List[float]] = None
    mp_vertical_flip: Optional[bool] = None
    mp_horizontal_flip: Optional[bool] = None
    mp_rotation_factor: Optional[List[float]] = None
    mp_zoom_factor: Optional[List[float]] = None
    mp_contrast_factor: Optional[List[float]] = None
    # ResNet
    mp_version: Optional[str] = None
    mp_pretrained: Optional[bool] = None
    # Train pipeline
    # AutoModel
    tp_project_name: str = field(default="auto_model")
    tp_max_trials: int = field(default=1)
    tp_directory: Optional[str] = None
    tp_objective: str = field(default="val_loss")
    tp_tuner: str = field(default="greedy")
    tp_overwrite: bool = field(default=False)
    tp_seed: Optional[int] = None
    tp_max_model_size: Optional[int] = None
    # AutoModel.fit()
    tp_batch_size: int = field(default=32)
    tp_epochs: Optional[int] = None
    tp_validation_split: float = field(default=0.2)

def main():
    parser = AutoArgumentParser((ResNetTrainerArguments))
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
    
    logger.info(f"{'-'*5} Start training {'-'*5}")
    trainer.train(inputs=trainer_args.inputs)
    
    train_summary = trainer.get_summary()
    logger.info(f"{'-'*5} Train summary {'-'*5}:\n{train_summary}")

if __name__=="__main__":
    main()