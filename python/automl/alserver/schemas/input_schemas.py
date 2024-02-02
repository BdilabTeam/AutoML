from pydantic import BaseModel, Field
from typing import Literal, IO, Dict, Any


class TrainingProjectBase(BaseModel):
    project_name: str = Field(description="项目名称")
    task_type: str = Field(description="任务类型")
    model_type: str= Field(description="模型 or 算法类型")
    # inputs: IO = Field(description="训练数据")
    inputs: str = Field(description="训练数据")

    class Config:
        arbitrary_types_allowed = True

class TrainingProjectCreate(TrainingProjectBase):
    max_trials: int = Field(description="最大试验次数", default=1, ge=1)
    tuner: Literal["greedy", "bayesian", "hyperband", "random"] = Field(description="超参数调优算法")
    training_params: Dict[str, Any] = Field(description="当前'模型'对应训练器的配置参数")

class CandidateModelSelect(BaseModel):
    task_type: str = Field(description="任务类型")
    task_desc: str = Field(description="任务需求描述")
    model_nums: int = Field(description="期望推荐的模型数量", default=1, ge=1)
