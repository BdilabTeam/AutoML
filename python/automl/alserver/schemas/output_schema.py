from pydantic import BaseModel as _BaseModel, Field
from typing import Dict, List, Any, Optional

class BaseModel(_BaseModel):
    class Config:
        extra = 'ignore'

class CandidateModel(BaseModel):
    id: Optional[str] = Field(description="ID of the model")
    reason: Optional[str] = Field(description="Reason for selecting this model")
    training_params: Optional[Dict[str, Any]] = Field(description="Model Training Configuration Parameters")
    
class CandidateModels(BaseModel):
    candidate_models: List[CandidateModel] = Field(description="候选模型列表, 每单项表示一个候选模型")

class JobInfo(BaseModel):
    job_name: Optional[str]
    job_status: Optional[Any]
    
class TrainingProjectInfo(BaseModel):
    id: Optional[int]
    project_name: Optional[str]
    task_type: Optional[str]
    model_type: Optional[str]
    training_params: Optional[Dict[str, Any]]
    job_info: Optional[JobInfo]