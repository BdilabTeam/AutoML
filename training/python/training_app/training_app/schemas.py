from pydantic import (
    BaseModel, 
    Field, 
    HttpUrl, 
    ConfigDict
)

from typing import Optional
from pydantic.dataclasses import dataclass


config = ConfigDict(
    title="Model config",
    from_attributes=True,
    protected_namespaces=("model_name_or_path")
)


# class TrainerArguments(BaseModel):
#     id: int = Field(default=0, title="训练参数ID")
#     data_file: str = Field(default=None, title="数据集文件路径")
#     model_name_or_path: str = Field(default=None, title="模型名称")
#     task_type: int = Field(default=None, title="任务类型")
    

class TrainingProjectCreate(BaseModel):
    name: Optional[str] = Field(description="项目名称", max_length=30, examples=["project1"])
    task_type: Optional[str] = Field(description="任务类型")
    is_automatic: Optional[bool] = Field(default=False, description="是否开启自动选择模型")
    model_name_or_path: Optional[Optional[str]] = Field(description="模型类型")
    

class TrainingProjectUpdate(TrainingProjectCreate):
    # id: Optional[int] = Field(title="项目ID")
    data_name_or_path: Optional[str] = Field(default="", description="已上传的数据文件路径")
    # host: Optional[str] = Field(default="", description="部署的knative虚拟主机名")


@dataclass(config=config) 
class TrainingProject(TrainingProjectCreate):
    id: Optional[int] = Field(title="项目ID")
    data_name_or_path: Optional[str] = Field(default="", description="已上传的数据文件路径")
    host: Optional[str] = Field(default="", description="部署的knative虚拟主机名")