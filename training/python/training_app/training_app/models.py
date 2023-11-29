from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.orm import (
    mapped_column,
    Mapped
)

from database import Base
from typing import Optional


class TrainingProject(Base):
    __tablename__ = "training_project"
    
    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(type_=String(30))
    task_type: Mapped[str] = mapped_column(type_=String(30))
    is_automatic: Mapped[bool] = mapped_column(type_=Boolean())
    model_name_or_path: Mapped[Optional[str]] = mapped_column(type_=String(30), doc="训练所用到的模型文件路径")
    data_name_or_path: Mapped[Optional[str]] = mapped_column(type_=String(50), doc="训练所用到的数据文件路径")
    host: Mapped[Optional[str]] = mapped_column(init=False, type_=String(50), doc="部署的knative虚拟主机名")
