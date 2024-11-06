from pydantic import BaseModel as _BaseModel, Field
from typing import Dict, List, Any, Optional, Literal

class BaseModel(_BaseModel):
    class Config:
        extra = 'ignore'

class CandidateModel(BaseModel):
    model_type: Optional[str] = Field(description="基础模型")
    reason: Optional[str] = Field(description="推荐原因")
    training_params: Optional[Dict[str, Any]] = Field(description="训练参数")
    
class CandidateModels(BaseModel):
    candidate_models: List[CandidateModel] = Field(description="候选模型列表, 每单项表示一个候选模型")
 
class ExperimentInfo(BaseModel):
    experiment_name: Optional[str]
    # task_type: Optional[str]
    # model_type: Optional[str]
    # training_params: Optional[Dict[str, Any]]
    # job_info: Optional[JobInfo]

class ExperimentCard(BaseModel):
    experiment_name: Optional[str] = Field(description="实验名称")
    task_type: Optional[str] = Field(description="任务类型")
    task_desc: Optional[str] = Field(description="任务描述")
    model_type: Optional[str] = Field(description="基础模型")
    experiment_status: Optional[str] = Field(description="实验状态")
   
class ExperimentCards(BaseModel):
    experiment_cards: List[ExperimentCard] = Field(description="实验卡片列表")

class Trial(BaseModel):
    trial_id: Optional[str] = Field(description="试验ID")
    trial_status: Optional[str] = Field(description="试验状态")
    default_metric: Optional[float] = Field(description="默认指标")
    best_step: Optional[int] = Field(description="最佳步骤")
    parameters: Optional[Dict[str, Any]] = Field(description="参数")
    model_graph_url: Optional[str] = Field(description="模型结构图路径")

class BestModel(BaseModel):
    history: Optional[Dict[str, Any]] = Field(description="训练历史")
    parameters: Optional[Dict[str, Any]] = Field(description="参数")
    model_graph_url: Optional[str] = Field(description="模型结构图路径")

class ExperimentOverview(BaseModel):
    experiment_name: Optional[str]  = Field(description="实验名称")
    experiment_status: Optional[str] = Field(description="实验状态")
    experiment_start_time: Optional[str] = Field(description="实验开始时间")
    experiment_completion_time: Optional[str] = Field(description="实验结束时间")
    experiment_duration_time: Optional[str] = Field(description="实验持续时间")
    # experiment_summary_url: Optional[str] = Field(description="实验总结")
    experiment_summary: Optional[Dict[str, Any]] = Field("实验总结")
    tuner: Optional[str] = Field(description="调优算法")
    max_trial_number: Optional[int] = Field(description="最大调优次数")
    trials: Optional[List[Trial]] = Field(description="调优试验列表详情")
    best_model: Optional[BestModel] = Field(description="最优模型详情")

class ModelRepository(BaseModel):
    models: Optional[List[str]] = Field(description="模型列表")

class EvaluateResponse(BaseModel):
    metrics: Optional[Dict[str, Any]] = Field(description="测试评估指标")
    # y_true: Optional[List] = Field(description="真实标签")
    # y_pred: Optional[List] = Field(description="预测标签")

class FileInfo(BaseModel):
    file_name: Optional[str] = Field(description="文件名称")
    file_path: Optional[str] = Field(description="文件路径")
    file_type: Literal["file", "directory"] = Field(description="文件类型")
    
class DatasetInfoResponse(BaseModel):
    file_infos: List[FileInfo] = Field(description="文件列表")
    