from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingArguments(object):
    task_type: str = field(
        default=None,
        metadata={
            "help": "The type name of the training task."
        }
    )
    
    model_type: str = field(
        default=None,
        metadata={
            "help": "The type name of the model."
        }
    )
    
    train_dir: str = field(
        default=None,
        metadata={
            "help": "The path/dir of model training data."
        }
    )
    
    # AutoModel config
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to store training results."
        }
    )
    
    tp_overwrite: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to override the output_dir."
        }
    )
    
    tp_project_name: Optional[str] = field(
        default="auto_model",
        metadata={
            "help": "The name of the training project."
        }
    )
    
    tp_max_trials: Optional[int] = field(
        default=5,
        metadata={
            "help": "Maximum number of trials."
        }
    )
    
    tp_objective: Optional[str] = field(
        default="val_loss",
        metadata={
            "help": "The goal of the hyperparameter tuning reference."
        }
    )
    
    tp_tuner: Optional[str] = field(
        default="greedy",
        metadata={
            "help": "Hyperparameter tuning algorithm used in training."
        }
    )
    
    # fit config
    batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": "Training data batch size."
        }
    )
    
    epochs: Optional[int] = field(
        default=500,
        metadata={
            "help": "Training round."
        }
    )
    
    validation_split: Optional[float] = field(
        default=0.2,
        metadata={
            "help": ""
        }
    )
    
    is_early_stop: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enable the training early stop feature"
        }
    )
    
    do_auto_feature_extract: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enable automatic feature extraction"
        }
    )
    
    do_auto_hyperparameter_tuning: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enable automatic hyperparameter tuning"
        }
    )