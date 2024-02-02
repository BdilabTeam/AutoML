from pydantic import BaseModel, Field
from typing import Dict, List, Any

class CandidateModel(BaseModel):
    id: str = Field(description="ID of the model")
    reason: str = Field(description="Reason for selecting this model")
    training_params: Dict[str, Any] = Field(description="Model Training Configuration Parameters")
    
class CandidateModels(BaseModel):
    candidate_models: List[CandidateModel] = Field(description="候选模型列表, 每单项表示一个候选模型")

class TrainingProjectInfo(BaseModel):
    pass