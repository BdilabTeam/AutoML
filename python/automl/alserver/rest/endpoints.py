from fastapi.responses import JSONResponse
from fastapi import Body, File, UploadFile, Form, Path

from ..handlers import DataPlane
from ..schemas import input_schemas, output_schemas
import json


class Endpoints(object):
    """
    Implementation of REST endpoints.
    These take care of the REST/HTTP-specific things and then delegate the
    business logic to the internal handlers.
    """
    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane
    
    # async def create_training_project(self, 
    #                          files: List[UploadFile] = File(description="Multiple files as UploadFile"), 
    #                          template_id: str = Form(),
    #                          template_parameters: str = Form()) -> JSONResponse:
    #        return JSONResponse(
    #         content="{'test': 'ok'}"
    #     )
    def create_training_project(self, training_project_vo: input_schemas.TrainingProjectCreate) -> JSONResponse:
        # 获取训练函数
        train_func = self._data_plane.get_train_func(model_type=training_project_vo.model_type)
        # 获取训练函数接口所需的参数字典
        # TODO 接收数据文件、存储数据文件、确定inputs字段值
        training_project_vo.training_params.update({
            'task_type': training_project_vo.task_type,
            'model_type': training_project_vo.model_type,
            'inputs': training_project_vo.inputs
        })
        # TODO 获取调度信息
        self._data_plane.create_training_job(
            name=training_project_vo.project_name,
            func=train_func,
            parameters=training_project_vo.training_params,
            host_ip='60.204.186.96'
        )
        # TODO 响应训练项目相关信息
        return JSONResponse(content='The training job was created successfully')
    
    def delete_training_project(self):
        pass
    
    def delete_training_job(self, training_job_name: str = Path()):
        self._data_plane.delete_training_job(name=training_job_name)
        return JSONResponse(content=f"The training job '{training_job_name}' was deleted successfully")

    async def get_candidate_models(self, candidate_model_select_vo: input_schemas.CandidateModelSelect = Body()) -> JSONResponse:
        # 获取候选模型
        models = await self._data_plane.aselect_models(
            user_input=candidate_model_select_vo.task_desc,
            task=candidate_model_select_vo.task_type,
            model_nums=candidate_model_select_vo.model_nums
        )
        candidate_models = []
        for model in models:
            # 获取候选模型对应的训练器配置参数
            trainer_id = candidate_model_select_vo.task_type + '/' + model.id
            training_params_dict = self._data_plane.get_training_params_dict(trainer_id=trainer_id)
            
            candidate_model = output_schemas.CandidateModel(
                id=str.lower(model.id),
                reason=model.reason,
                training_params=training_params_dict
            )
            candidate_models.append(candidate_model)
        return output_schemas.CandidateModels(candidate_models=candidate_models)
        
    
    def delete_training_project(self) -> JSONResponse:
        pass

    def update_training_project(self) -> JSONResponse:
        pass
    
    def get_training_project(self) -> JSONResponse:
        pass

    def get_training_projects(self) -> JSONResponse:
        pass

    def start_training_job(self) -> JSONResponse:
        pass
    
    def stop_training_job(self) -> JSONResponse:
        pass
    
    def get_training_job_info(self) -> JSONResponse:
        pass
        
    async def get_monitor_info(self) -> JSONResponse:
        return JSONResponse(
        content="{'test': 'ok'}"
    )