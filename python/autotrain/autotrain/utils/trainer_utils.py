import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Optional, List, Dict

from .configuration_utils import BaseTrainerConfig

@dataclass
class BestModelTracker:
    history: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    model_graph_path: Optional[str]

@dataclass 
class Trial:
    trial_id: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    best_step: int
    status: str
    model_graph_path: Optional[str]
    message: Any

@dataclass 
class TrialsTracker:
    trials: List[Trial]

@dataclass 
class ConfigTracker:
    label2ids: Dict
    id2labels: Dict
    
@dataclass
class TrainerTracker(object):
    best_model_tracker: Optional[BestModelTracker]
    trials_tracker: Optional[TrialsTracker]
    config_tracker: Optional[ConfigTracker]

class BaseTrainer(object):
    
    def __init__(self, config: BaseTrainerConfig) -> None:
        self.config = config
    
    def train(self, inputs: Any, *args: Any, **kwds: Any):
        raise NotImplementedError

    def save_summary(self, tracker: TrainerTracker):
        with open(os.path.join(self.config.tp_directory, self.config.tp_project_name, 'summary.json'), 'w') as f:
            json.dump(asdict(tracker), f)
            
    def get_summary(self) -> dict:
        with open(os.path.join(self.config.tp_directory, self.config.tp_project_name, 'summary.json'), "r") as f:
            return json.loads(f.read())
    

