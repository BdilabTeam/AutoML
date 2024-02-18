from sqlalchemy import String
from sqlalchemy.orm import mapped_column, Mapped

from .base import Base


class Experiment(Base):
    __tablename__ = "experiment"
    
    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    experiment_name: Mapped[str] = mapped_column(type_=String(50))
    task_type: Mapped[str] = mapped_column(type_=String(50))
    task_desc: Mapped[str] = mapped_column(type_=String(150), doc="任务描述信息")
    model_type: Mapped[str] = mapped_column(type_=String(50))
    experiment_job_name: Mapped[str] = mapped_column(init=False, nullable=True, type_=String(50))
    workspace_dir: Mapped[str] = mapped_column(init=False, type_=String(100), nullable=True, doc="训练项目工作路径")
    training_params_file_path: Mapped[str] = mapped_column(init=False, type_=String(100), nullable=True, doc="训练参数配置文件路径")
    model_dir: Mapped[str] = mapped_column(init=False, type_=String(100), nullable=True, doc="模型文件路径")
    data_dir: Mapped[str] = mapped_column(init=False, type_=String(100), nullable=True, doc="训练数据文件路径")
    virtual_host: Mapped[str] = mapped_column(init=False, type_=String(255), nullable=True, doc="虚拟主机名称")
    