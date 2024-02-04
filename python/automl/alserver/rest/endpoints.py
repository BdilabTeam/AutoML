import json
from pathlib import Path
from typing import List, Dict, Any, Union, Literal
from fastapi.responses import JSONResponse
from fastapi import Body, File, UploadFile, Form

from ..handlers import DataPlane
from ..schemas import input_schema, output_schema
from ..errors import DataFormatError


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
    
    def create_training_project(
        self, 
        files: List[UploadFile] = File(description="Multiple files as UploadFile"),
        # training_project_vo: input_schema.TrainingProjectCreate = Form(),
        project_name: str = Form(...),
        task_type: Literal["structured-data-classification", "structured-data-regression", "image-classification", "image-regression"] = Form(...),
        model_type: Literal["densenet", "resnet"] = Form(...),
        max_trials: int = Form(...),
        tuner: Literal["greedy", "bayesian", "hyperband", "random"] = Form(...),
        training_params: Union[Dict, Any] = Form(...)
    ) -> output_schema.TrainingProjectInfo:
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
        elif len(path_parts) == 2:
            for file in files:
                # Check the depth of the folder. The length should be less than or equal to 2.
                if len(path_parts) != 2:
                    raise DataFormatError("图片数据文件格式错误")
                if not file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    raise DataFormatError(f"图片格式错误, 目前仅支持扩展名必须为:[.jpg, .jpeg, .png, .gif, .bmp]的图片")
            file_type = 'image_folder'
        else:
            raise DataFormatError("数据文件格式错误")
        
        # 合并训练参数
        training_params.update(
            {
                "tp_max_trials": max_trials,
                "tp_tuner": tuner
            }
        )
        # TODO 获取调度信息
        host_ip = '60.204.186.96'
        
        training_project_info = self._data_plane.create_training_project(
            name=project_name,
            task_type=task_type,
            model_type=model_type,
            training_params=training_params,
            file_type=file_type,
            files=files,
            host_ip=host_ip
        )
        return training_project_info
    
    def delete_training_project(self, training_project_id: int = Path()) -> JSONResponse:
        self._data_plane.delete_training_project(training_project_id=training_project_id)
        return JSONResponse(content=f'Success to delete  {training_project_id} traininig project')
    
    def get_training_project_info(self, training_project_id: int = Path()) -> output_schema.TrainingProjectInfo:
        training_project_info = self._data_plane.get_training_project_info(training_project_id=training_project_id)
        return training_project_info
    
    def delete_training_job(self, training_job_name: str = Path()):
        self._data_plane.delete_training_job(name=training_job_name)
        return JSONResponse(content=f"The training job '{training_job_name}' was deleted successfully")

    def start_training_job(self) -> JSONResponse:
        pass
    
    def update_training_project(self) -> JSONResponse:
        pass
    
    def get_training_project(self) -> JSONResponse:
        pass

    def get_training_projects(self) -> JSONResponse:
        pass

    
    def get_training_job_info(self) -> JSONResponse:
        pass
        
    async def get_monitor_info(self) -> JSONResponse:
        return JSONResponse(
        content="{'test': 'ok'}"
    )