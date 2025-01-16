from sqlalchemy import String
from sqlalchemy.orm import mapped_column, Mapped

from .base import Base


class Experiment(Base):
    __tablename__ = "experiment"
    __table_args__ = {
        'mysql_charset': 'utf8mb4',
        'mysql_collate': 'utf8mb4_unicode_ci'
    }    
    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    experiment_name: Mapped[str] = mapped_column(type_=String(50))
    task_type: Mapped[str] = mapped_column(type_=String(50))
    task_desc: Mapped[str] = mapped_column(type_=String(150), doc="任务描述信息", nullable=True)
    model_type: Mapped[str] = mapped_column(type_=String(50))
    tuner_type: Mapped[str] = mapped_column(init=False, nullable=True, type_=String(50))
    workspace_dir: Mapped[str] = mapped_column(init=False, type_=String(100), nullable=True, doc="实验工作目录")
    training_params: Mapped[str] = mapped_column(init=False, type_=String(5000), nullable=True, doc="实验训练参数")