import os
from typing import Dict, Any

from autotrain.trainers.auto import AutoConfig, AutoTrainer
from autotrain.utils.logging import get_logger

logger = get_logger(__name__)

def train_resnet(trainer_args: Dict[str, Any]):
    trainer_id = os.path.join(trainer_args.task_type, trainer_args.model_type)
    config = AutoConfig.from_repository(trainer_id=trainer_id)

    for key, value in trainer_args.items():
        if key in ['model_type', 'task_type', 'trainer_class_name']:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    
    trainer = AutoTrainer.from_config(config=config)
    logger.info(f"{'-'*5} Start training {'-'*5}")
    output = trainer.train(inputs=trainer_args.inputs)
    logger.info(f"{'-'*5} Training history {'-'*5}")
    print(output)