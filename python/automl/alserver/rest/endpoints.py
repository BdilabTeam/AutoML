import json
from pathlib import Path
from typing import List, Dict, Any, Union, Literal
from fastapi.responses import JSONResponse, Response
from fastapi import Body, File, UploadFile, Form, WebSocket

from ..handlers import DataPlane
from ..schemas import input_schema, output_schema
from ..errors import DataFormatError, WebSocketQueryParamError


class Endpoints(object):
    """
    Implementation of REST endpoints.
    These take care of the REST/HTTP-specific things and then delegate the
    business logic to the internal handlers.
    """
    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane
    
    async def get_candidate_models(self, candidate_model_select_vo: input_schema.CandidateModelSelect = Body()) -> output_schema.CandidateModels:
        # 获取候选模型
        candidate_models = await self._data_plane.aselect_models(
            user_input=candidate_model_select_vo.task_desc,
            task_type=candidate_model_select_vo.task_type,
            model_nums=candidate_model_select_vo.model_nums
        )
        return candidate_models
    
    async def create_experiment(
        self, 
        experiment_name: str = Form(description="实验名称", regex="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾"),
        task_type: Literal["structured-data-classification", "structured-data-regression", "image-classification", "image-regression"] = Form(description="任务类型"),
        task_desc: str = Form(max_length=150, example="钢材淬透性预测", description="任务描述"),
        model_type: Literal["densenet", "resnet", "xception", "convnet", "yolov8"] = Form(description="基础模型"),
        files: List[UploadFile] = File(description="上传单文件或文件夹"),
        tp_max_trials: int = Form(ge=1, description="最大试验输"),
        tp_tuner: Literal["greedy", "bayesian", "hyperband", "random"] = Form(description="参数调优算法"),
        training_params: Union[Dict, Any] = Form(description="训练参数")
    ) -> output_schema.ExperimentInfo:
        # 手动检查training_params字段值是否为dict格式
        if not isinstance(training_params, dict):
            try:
                training_params = json.loads(training_params)
            except (TypeError, ValueError):
                raise DataFormatError(f"training_params字段值错误, 期望为dict格式")
        
        if len(files) == 0:
            raise DataFormatError("数据文件不能为空")
        
        # 检查文件类型
        path_parts = Path(files[0].filename).parts
        if len(path_parts) == 1 and len(files) == 1:
            if files[0].filename.endswith('csv'):
                file_type = 'csv'
            else:
                raise DataFormatError(f"仅[csv]扩展文件类型")
        elif len(path_parts) == 3:
            file_type = 'image_folder'
            for file in files:
                # Check the depth of the folder. The length should be less than or equal to 2.
                if len(path_parts) != 3:
                    raise DataFormatError("图片数据文件格式错误")
                
                if file.filename.endswith('.json'):
                    file_type = 'marked_image_folder'
                elif not file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    raise DataFormatError(f"图片格式错误, 扩展名必须为:[.jpg, .jpeg, .png, .gif, .bmp]")
            
        else:
            # TODO 适配yolo系列模型数据格式
            # raise DataFormatError("数据文件格式错误")
            file_type = 'image_folder'

        host_ip = None
        # host_ip = "60.204.186.96"
        experiment_info = await self._data_plane.create_experiment(
            experiment_name=experiment_name,
            task_type=task_type,
            task_desc=task_desc,
            model_type=model_type,
            training_params=training_params,
            file_type=file_type,
            files=files,
            host_ip=host_ip,
            tp_max_trials=tp_max_trials,
            tp_tuner=tp_tuner
        )
        return experiment_info
    
    def patch_experiment(
        self, 
        experiment_name: str = Form(description="实验名称", regex="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾"),
        task_desc: str = Form(max_length=150, example="钢材淬透性预测", description="任务描述"),
        # files: List[UploadFile] = File(description="上传单文件或文件夹"),
        training_params: Union[Dict, Any] = Form(description="训练参数")
    ) -> output_schema.ExperimentInfo:
        # 手动检查training_params字段值是否为dict格式
        if not isinstance(training_params, dict):
            try:
                training_params = json.loads(training_params)
            except (TypeError, ValueError):
                raise DataFormatError(f"training_params字段值错误, 期望为dict格式")
            
        return self._data_plane.patch_experiment(
            experiment_name=experiment_name,
            task_desc=task_desc,
            # files=files,
            training_params=training_params
        )
    
    def delete_experiment(self, experiment_name: str = Path(title = "实验名称", description = "实验名称")) -> JSONResponse:
        self._data_plane.delete_experiment(experiment_name=experiment_name)
        return JSONResponse(content=f'Success to delete {experiment_name}')
    
    def get_experiment_overview(self, experiment_name: str = Path(title = "实验名称", description="实验名称")) -> output_schema.ExperimentOverview:
        experiment_overview = self._data_plane.get_experiment_overview(experiment_name=experiment_name)
        return experiment_overview
    
    def get_experiment_cards(self) -> output_schema.ExperimentCards:
        experiment_cards = self._data_plane.get_experiment_cards()
        return experiment_cards
    
    async def get_experiment_job_logs(self, websocket: WebSocket):
        # 处理 connect 消息
        await websocket.accept()
        experiment_job_name = websocket.query_params.get("experiment_job_name", None)
        if not experiment_job_name:
            raise WebSocketQueryParamError("Expect to include the 'experiment_job_name' request parameter")
        await self._data_plane.get_experiment_logs(experiment_name=experiment_job_name, websocket=websocket)
        # await websocket.close(reason="Completed")
    
    def get_model_repository_info(self) -> output_schema.ModelRepository:
        model_repository = self._data_plane.get_model_repository()
        return model_repository
    
    def evaluate_experiment(
        self,
        experiment_name: str = Form(description="实验名称", regex="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾"),
        task_type: Literal["structured-data-classification", "structured-data-regression", "image-classification", "image-regression"] = Form(description="任务类型"),
        files: List[UploadFile] = File(description="上传单文件或文件夹"),
    ) -> output_schema.EvaluateResponse:
        if len(files) == 0:
            raise DataFormatError("数据文件不能为空")
        # 检查文件类型
        path_parts = Path(files[0].filename).parts
        if len(path_parts) == 1 and len(files) == 1:
            if files[0].filename.endswith('csv'):
                file_type = 'csv'
            else:
                raise DataFormatError(f"仅[csv]扩展文件类型")
        elif len(path_parts) == 3:
            for file in files:
                # Check the depth of the folder. The length should be less than or equal to 2.
                if len(path_parts) != 3:
                    raise DataFormatError("图片数据文件格式错误")
                if not file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    raise DataFormatError(f"图片格式错误, 扩展名必须为:[.jpg, .jpeg, .png, .gif, .bmp]")
            file_type = 'image_folder'
        else:
            # TODO 适配yolo系列模型数据格式
            # raise DataFormatError("数据文件格式错误")
            file_type = 'image_folder'
        
        metrics = self._data_plane.evaluate_experiment(
            experiment_name=experiment_name,
            task_type=task_type,
            file_type=file_type,
            files=files
        )
        return output_schema.EvaluateResponse(metrics=metrics)
    
    def export_best_model(self, experiment_name: str = Path(title = "实验名称", description="实验名称")):
        model_zipf_bytes = self._data_plane.export_best_model(experiment_name=experiment_name)
        # 返回 zip 文件，设置正确的 Content-Disposition 头部以触发下载
        return Response(
            content=model_zipf_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={experiment_name}.zip"
            }
        )
    
    def get_dataset_overview(self, experiment_name: str = Path(title = "实验名称", description="实验名称")):
        pass
    
    def get_minio_url(self):
        minio_url = self._data_plane.get_minio_url()
        return minio_url
    
    def get_data_annotation_platform_url(self):
        data_annotation_platform_url = self._data_plane.get_data_annotation_platform_url()
        return data_annotation_platform_url
    
    def get_task_types(self):
        return self._data_plane.get_task_types()