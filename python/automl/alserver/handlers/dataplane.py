import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Literal, List, TypeVar

from ..settings import Settings
from ..databases.mysql import MySQLClient
from ..errors import (
    MySQLNotExistError, 
    SelectModelError, 
    DeleteJobError,
    CreateJobError,
    GetJobInfoError,
    GetTrainingParamsError,
    SaveTrainingParamsError,
    TrainingProjectNotExistError
)
from ..operators import TrainingClient
from ..utils import dataplane_utils
from ..schemas import input_schema, output_schema

UploadFile = TypeVar('UploadFile')


class DataPlane:
    """
    Internal implementation of handlers, used by REST servers.
    """
    def __init__(self, settings: Settings):
        
        if settings.mysql_enabled:
            self._mysql_client = MySQLClient(settings)

        if settings.kubernetes_enabled:
            self._training_client = TrainingClient(
                config_file=settings.kube_config_file, 
            )
            
        if settings.model_selection_enabled:
            from autoselect import (
                ModelSelection, ModelSelectionSettings
            )
            
            model_selection_settings = ModelSelectionSettings(
                prompt_template_file_path=settings.prompt_template_file_path,
                model_metadata_file_path=settings.model_metadata_file_path
            )
            self._model_selection_service = ModelSelection(settings=model_selection_settings)
            
        
        if settings.monitor_enabled:
            from autoschedule import ResourceMonitor
            import threading
            
            self._resource_monitor_service = ResourceMonitor(
                host_info_file_path=self._settings.host_info_file_path
            )
            # 守护线程
            threading.Thread(target=self._resource_monitor_service.start(), daemon=True).start()
            
        self._settings = settings
    
    def get_session(self):
        """Provide database session"""
        if not hasattr(self, '_mysql_client'):
            raise MySQLNotExistError("No available MySQL database server")
        
        return self._mysql_client.get_session()
    
    @staticmethod
    def transactional(func):
        def wrapper(self, *args, **kwargs):
            session = self.get_session()
            try:
                transaction = session.begin()
                result = func(self, session=session, *args, **kwargs)
                transaction.commit()
                return result
            except:
                transaction.rollback()
                raise
            finally:
                session.close()

        return wrapper
    
    async def aselect_models(
        self, 
        user_input: str, 
        task_type: str, 
        model_nums: int = 1
    ) -> output_schema.CandidateModels:
        from autoselect import LLMFactory, ModelSelectionLLMSettings, OutputFixingLLMSettings
        
        model_selection_llm_settings = ModelSelectionLLMSettings(
            env_file_path=self._settings.env_file_path,
            temperature=0
        )
        model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        
        output_fixing_llm_settings = OutputFixingLLMSettings(
            env_file_path=self._settings.env_file_path,
            temperature=0
        )
        output_fixing_llm = LLMFactory.get_output_fixing_llm(output_fixing_llm_settings)
        
        try:
            models = await self._model_selection_service.aselect_model(
                user_input=user_input,
                task=task_type,
                top_k=model_nums * 2,
                model_nums=model_nums,
                model_selection_llm=model_selection_llm,
                output_fixing_llm=output_fixing_llm,
                description_length=100
            )
        except Exception as e:
            raise SelectModelError(f"Failed to select the candidate model, for a specific reason: {e}")
        
        candidate_models = []
        for model in models:
            # 获取候选模型对应的训练器配置参数
            try:
                training_params_dict = dataplane_utils.get_training_params_dict(task_type=task_type, model_type=model.id)
            except Exception as e:
                raise GetTrainingParamsError(f"Failed to get the train parameters, for a specific reason: {e}")
            candidate_model = output_schema.CandidateModel(
                id=str.lower(model.id),
                reason=model.reason,
                training_params=training_params_dict
            )
            candidate_models.append(candidate_model)
        return output_schema.CandidateModels(candidate_models=candidate_models)
    
    @transactional
    def create_training_project(
        self, 
        name: str,
        task_type: str,
        model_type: str,
        training_params: Dict[str, Any],
        file_type: Literal['csv', 'image_folder'],
        files: List[UploadFile],
        host_ip: str,
        **kwargs
    ) -> output_schema.TrainingProjectInfo:
        from autotrain import AutoTrainFunc
        from ..models.taining_project_model import TrainingProject
        from ..cruds.training_project_crud import create_training_project
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        try:
            # 数据库-创建训练项目
            training_project = TrainingProject(project_name=name, task_type=task_type, model_type=model_type)
            training_project = create_training_project(session=session, training_project=training_project)
            training_project.job_name = dataplane_utils.get_training_job_name(training_project.id, training_project.project_name)
            workspace_dir = dataplane_utils.generate_training_project_workspace_dir(worspace_name=str(training_project.id))
            training_project.workspace_dir = workspace_dir
            # 解析、存储数据
            data_dir = os.path.join(workspace_dir, 'datasets')
            # training_project.data_dir = data_dir
            if file_type == 'csv':
                file_path = Path(data_dir, files[0].filename)
                file_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
                with file_path.open("wb") as buffer:
                    shutil.copyfileobj(files[0].file, buffer)
                inputs = Path(dataplane_utils.DATA_DIR_IN_CONTAINER, files[0].filename).__str__()
            elif file_type == 'image_folder':
                for file in files:
                    # path_parts = Path(file.filename).parts
                    # file_path = Path(data_dir, dataplane_utils.IMAGE_FOLDER_NAME, *path_parts)
                    file_path = Path(data_dir, dataplane_utils.IMAGE_FOLDER_NAME, file.filename)
                    file_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
                    with file_path.open("wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                inputs = Path(dataplane_utils.DATA_DIR_IN_CONTAINER, dataplane_utils.IMAGE_FOLDER_NAME).__str__()
            else:
                raise ValueError
                    
            # 获取训练函数
            train_func = AutoTrainFunc.from_model_type(model_type)
            # 更新训练参数
            training_params.update(
                {
                    'tp_project_name': name,
                    'task_type': task_type,
                    'model_type': model_type,
                    'inputs': inputs,
                    'tp_directory': dataplane_utils.WORKSPACE_DIR_IN_CONTAINER
                }
            )
            
            # 训练参数字典存储为json文件
            training_params_file_path = os.path.join(workspace_dir, dataplane_utils.TRAINING_PARAMETERS_FILE_NAME)
            training_project.trainig_params_file_path = training_params_file_path
            try:
                dataplane_utils.save_dict_to_json_file(data=training_params, json_file=training_params_file_path)
            except Exception as e:
                raise SaveTrainingParamsError(f"Failed to save the training parameters, for a specific reason: {e}")
            
            # 生成job名称
            job_name = dataplane_utils.get_training_job_name(
                training_project_id=training_project.id, 
                training_project_name=training_project.project_name
            )
            # 发布训练作业
            try:
                self._training_client.create_tfjob_from_func(
                    name=job_name,
                    func=train_func,
                    parameters=training_params,
                    base_image=self._settings.base_image,
                    namespace=self._settings.namespcae,
                    num_worker_replicas=1,
                    host_ip=host_ip,
                    external_workspace_dir=workspace_dir,
                )
            except Exception as e:
                raise CreateJobError(f"Failed to create a training job '{name}', for a specific reason: {e}")
            
            return output_schema.TrainingProjectInfo(
                id=training_project.id,
                project_name=training_project.project_name,
                task_type=training_project.task_type,
                model_type=training_project.model_type,
            )
        except:
            dataplane_utils.remove_workspace_dir(workspace_dir=workspace_dir)
            self.delete_training_job(name=job_name)
            raise
    
    @transactional
    def delete_training_project(self, training_project_id: int, **kwargs):
        from ..cruds.training_project_crud import delete_training_project, get_training_project
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        try: 
            training_project = get_training_project(session=session, training_project_id=training_project_id)
        except:    
            raise TrainingProjectNotExistError(f"ID: {training_project_id} for training project does not exist.")
        
        delete_training_project(session=session, training_project_id=training_project_id)
        self.delete_training_job(name=training_project.job_name)
        dataplane_utils.remove_workspace_dir(workspace_dir=training_project.workspace_dir)
            
    
    def get_training_project_info(self, training_project_id: int) -> output_schema.TrainingProjectInfo:
        from ..cruds.training_project_crud import get_training_project
        
        session = self.get_session()
        try: 
            training_project = get_training_project(session=session, training_project_id=training_project_id)
        except:    
            raise TrainingProjectNotExistError(f"ID: {training_project_id} for training project does not exist.")
        
        job_name = training_project.job_name
        # try:
        #     job_status = self._training_client.get_tfjob(name=job_name, namespace=self._settings.namespcae).status
        # except Exception as e:
        #     raise GetJobInfoError(f"Failed to get training job '{job_name}' status information, for a specific reason: {e}")
        
        # 获取训练参数
        file_path = training_project.trainig_params_file_path
        with open(file_path, 'r') as file:
            training_params = json.load(file)
        
        return output_schema.TrainingProjectInfo(
            id=training_project_id,
            project_name=training_project.project_name,
            task_type=training_project.task_type,
            model_type=training_project.model_type,
            training_params=training_params,
            job_info=output_schema.JobInfo(
                job_name=job_name,
                # job_status=job_status
            )
        )
    
    def delete_training_job(self, name: str):
        try:
            self._training_client.delete_tfjob(
                name=name, 
                namespace=self._settings.namespcae
            )
        except Exception as e:
            raise DeleteJobError(f"Failed to delete a training job '{name}', for a specific reason: {e}")
    
    def is_training_job_succeeded(self, name: str):
        """Verify if job is succeeded"""
        try:
            return self._training_client.is_job_succeeded(name=name, namespace=self._settings.namespcae)
        except Exception as e:
            raise GetJobInfoError(f"Failed to get training job '{name}' status information, for a specific reason: {e}")
    
    def get_training_job_conditions(self, name: str):
        """Get the Training Job conditions."""
        try:
            conditions = self._training_client.get_job_conditions(
                name=name,
                namespace=self._settings.namespcae
            )
            return conditions
        except Exception as e:
            raise GetJobInfoError(f"Failed to get training job '{name}' status information, for a specific reason: {e}")
    
    def get_training_job_logs(self, name: str):
        return self._training_client.get_job_logs(
            name=name,
            namespace=self._settings.namespcae
        )
    
    def get_gpu_and_host(self, threshold):
        return self._resource_monitor_service.get_gpu_and_host(threshold=threshold)