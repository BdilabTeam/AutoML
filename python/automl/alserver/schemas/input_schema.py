from pydantic import BaseModel, Field
from typing import Literal, IO, Dict, Any, Union


class ExperimentBase(BaseModel):
    experiment_name: str = Field(description="项目名称")
    task_type: Literal["structured-data-classification", "structured-data-regression", "image-classification", "image-regression"] = Field(description="任务类型")
    model_type: Literal["densenet", "resnet", "xception", "convnet", "yolov8"]= Field(description="模型 or 算法类型")
    files: IO = Field(description="训练数据")

    class Config:
        arbitrary_types_allowed = True

class ExperimentCreate(ExperimentBase):
    max_trials: int = Field(description="最大试验次数", default=1, ge=1)
    tuner: Literal["greedy", "bayesian", "hyperband", "random"] = Field(description="超参数调优算法")
    training_params: Union[Dict[str, Any], Any] = Field(description="当前'模型'对应训练器的配置参数")

class CandidateModelSelect(BaseModel):
    task_type: Literal["structured-data-classification", "structured-data-regression", "image-classification", "image-regression"] = Field(description="任务类型")
    task_desc: str = Field(description="任务需求描述")
    model_nums: int = Field(description="期望推荐的模型数量", default=1, ge=1)
