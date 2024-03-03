from pydantic import BaseModel as _BaseModel, Field
from typing import Dict, List, Any, Optional

class BaseModel(_BaseModel):
    class Config:
        extra = 'ignore'

class CandidateModel(BaseModel):
    model_type: Optional[str] = Field(description="ID of the model")
    reason: Optional[str] = Field(description="Reason for selecting this model")
    training_params: Optional[Dict[str, Any]] = Field(description="Model Training Configuration Parameters")
    
class CandidateModels(BaseModel):
    candidate_models: List[CandidateModel] = Field(description="候选模型列表, 每单项表示一个候选模型")

class JobInfo(BaseModel):
    experiment_job_name: Optional[str]
    experiment_job_status: Optional[Dict[str, Any]]
 
class ExperimentInfo(BaseModel):
    experiment_name: Optional[str]
    task_type: Optional[str]
    model_type: Optional[str]
    training_params: Optional[Dict[str, Any]]
    job_info: Optional[JobInfo]

class ExperimentCard(BaseModel):
    experiment_name: Optional[str]
    task_type: Optional[str]
    task_desc: Optional[str]
    model_type: Optional[str]
   
class ExperimentCards(BaseModel):
    experiment_cards: List[ExperimentCard]

class Trial(BaseModel):
    trial_id: Optional[str]
    trial_status: Optional[str]
    default_metric: Optional[float]
    best_step: Optional[int]
    parameters: Optional[Dict[str, Any]]
    model_graph_url: Optional[str]

class BestModel(BaseModel):
    history: Optional[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]]
    model_graph_url: Optional[str]

class ExperimentOverview(BaseModel):
    experiment_name: Optional[str]
    experiment_status: Optional[str]
    experiment_start_time: Optional[str]
    experiment_completion_time: Optional[str]
    experiment_duration_time: Optional[str]
    experiment_summary_url: Optional[str]
    tuner: Optional[str]
    max_trial_number: Optional[int]
    trials: Optional[List[Trial]]
    best_model: Optional[BestModel]

class ModelRepository(BaseModel):
    models: Optional[List[str]]
