def train_densenet(trainer_args: dict):
    import os
    import pandas as pd
    from autotrain.trainers.auto import AutoConfig, AutoTrainer, AutoFeatureExtractor
    from autotrain.utils.logging import get_logger
    from autotrain.utils.generic_utils import upload_dir_to_minio

    logger = get_logger(__name__)
    
    task_type = trainer_args.pop('task_type', None)
    model_type = trainer_args.pop('model_type', None)
    inputs = trainer_args.pop('inputs', None)
    minio_config = trainer_args.pop("minio_config", None)
    experiment_name = trainer_args.pop("experiment_name", None)
    
    trainer_id = os.path.join(task_type, model_type)
    config = AutoConfig.from_repository(trainer_id=trainer_id)

    for key, value in trainer_args.items():
        if key in ['model_type', 'task_type', 'trainer_class_name']:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    
    trainer = AutoTrainer.from_config(config=config)
    if config.dp_enable_auto_feature_extract:
        feature_extractor = AutoFeatureExtractor.from_config(config)
        feature_extract_output = feature_extractor.extract(
            inputs=inputs,
            trainer=trainer, 
        )
        
        logger.info(f"{'-'*5} Feature extraction history {'-'*5}")
        logger.info(f"{'*'*15}_Best Feature Index:\n{feature_extract_output.best_feature_index}")
        
        logger.info(f"Saving the extracted datasets.")
        best_feature_index = feature_extract_output.best_feature_index
        datasets = pd.read_csv(inputs)
        best_feature_index.append(-1)   # Add label index
        extracted_datasets = datasets.iloc[:, best_feature_index]
        extracted_file_name = '-'.join(['extracted', os.path.basename(inputs)])
        parent_dir = os.path.dirname(inputs)
        extracted_file_path = os.path.join(parent_dir, extracted_file_name)
        extracted_datasets.to_csv(extracted_file_path, index=False)
        inputs = extracted_file_path
    
    logger.info(f"{'-'*5} Start training {'-'*5}")
    trainer.train(inputs=inputs)
    
    train_summary = trainer.get_summary()
    logger.info(f"{'-'*5} Train summary {'-'*5}:\n{train_summary}")
    
    if minio_config:
        from minio import Minio
        
        minio_endpoint = minio_config.get("minio_endpoint")
        minio_access_key = minio_config.get("minio_access_key")
        minio_secret_key = minio_config.get("minio_secret_key")
        
        if not minio_endpoint or not minio_access_key or not minio_secret_key:
            logger.error(
                f"If you want to create minio client, you must specify the following key words: minio_endpoint、access_key、secret_key\
                    currently, the endpoint is {minio_endpoint}, the access_key is {minio_access_key}, the secret_key is {minio_secret_key}"
            )
            return
                
        minio_client = Minio(
            endpoint=minio_endpoint, 
            access_key=minio_access_key, 
            secret_key=minio_secret_key, 
            secure=False
        )
        upload_dir_to_minio(
            client=minio_client,
            bucket_name="automl",
            dir_path=os.path.join(trainer_args.get("tp_directory"), trainer_args.get("tp_project_name")),
            prefix=f"{experiment_name}/models"
        )
        
        