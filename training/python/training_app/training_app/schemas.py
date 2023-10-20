from pydantic import (
    BaseModel, 
    Field, 
    HttpUrl, 
    ConfigDict
)

from typing import Optional
from dataclasses import dataclass


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
    

class TrainingProjectBase(BaseModel):
    name: Optional[str] = Field(description="项目名称", max_length=30, examples=["project1"])
    task_type: Optional[str] = Field(description="任务类型")
    is_automatic: Optional[bool] = Field(default=False, description="是否开启自动选择模型")
    model_name_or_path: Optional[Optional[str]] = Field(description="模型类型")
    

class TrainingProjectCreate(TrainingProjectBase):
    pass


@dataclass(config=config)     
class TrainingProject(TrainingProjectBase):
    id: Optional[int] = Field(title="项目ID")
    data_name_or_path: Optional[str] = Field(default="", description="已上传的数据文件路径")
    
    # 使用 Pydantic 的 orm_mode
    # Pydantic orm_mode 将告诉 Pydantic模型读取数据，即它不是一个dict，而是一个 ORM 模型（或任何其他具有属性的任意对象）。
    # class Config():
    #     from_attributes = True
    #     # orm_mode = True