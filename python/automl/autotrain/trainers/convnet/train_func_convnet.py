def train_convnet(trainer_args: dict):
    import os
    from autotrain.trainers.auto import AutoConfig, AutoTrainer
    from autotrain.utils.logging import get_logger

    logger = get_logger(__name__)
    task_type = trainer_args.pop('task_type', None)
    model_type = trainer_args.pop('model_type', None)
    inputs = trainer_args.pop('inputs', None)
    
    trainer_id = os.path.join(task_type, model_type)
    config = AutoConfig.from_repository(trainer_id=trainer_id)

    for key, value in trainer_args.items():
        if key in ['model_type', 'task_type', 'trainer_class_name']:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    
    trainer = AutoTrainer.from_config(config=config)
    
    logger.info(f"{'-'*5} Start training {'-'*5}")
    trainer.train(inputs=inputs)
    
    train_summary = trainer.get_summary()
    logger.info(f"{'-'*5} Train summary {'-'*5}:\n{train_summary}")