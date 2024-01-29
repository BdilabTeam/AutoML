from pydantic import BaseModel, Field
from typing import Literal, IO


class TrainingProjectBase(BaseModel):
    project_name: str = Field(description="项目名称")
    task_type: str = Field(description="任务类型")
    model_type: str= Field(description="模型 or 算法类型")
    max_trials: int = Field(description="最大试验次数", default=1, ge=1)
    tuner: Literal["greedy", "bayesian", "hyperband", "random"] = Field(description="超参数调优算法")
    inputs: IO = Field(description="训练数据")

    class Config:
        arbitrary_types_allowed = True

class TrainingProjectCreate(TrainingProjectBase):
    pass