from typing import Optional, Dict, Any, Callable
from functools import partial

from ..settings import Settings
from ..databases.mysql import MySQLServer
from ..errors import (
    MySQLServerNotExistError, 
    SelectModelError, 
    DeleteJobError,
    CreateJobError,
    GetJobInfoError
)
from ..operators import TrainingClient


class DataPlane:
    """
    Internal implementation of handlers, used by REST servers.
    """
    def __init__(self, settings: Settings):
        
        if settings.mysql_enabled:
            self._mysql_server = MySQLServer(settings).start()

        if settings.kubernetes_enabled:
            self._training_client = TrainingClient(
                config_file=settings.kube_config_file, 
            )
            
        if settings.model_selection_enabled:
            from autoselect import (
                LLMFactory, ModelSelection, 
                ModelSelectionSettings, ModelSelectionLLMSettings
            )
            
            model_selection_settings = ModelSelectionSettings(
                prompt_template_file_path=settings.prompt_template_file_path,
                model_metadata_file_path=settings.model_metadata_file_path
            )
            model_selection = ModelSelection(settings=model_selection_settings)
            
            model_selection_llm_settings = ModelSelectionLLMSettings(
                env_file_path=settings.env_file_path,
                temperature=0.5
            )
            model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
            
            self.aselect = partial(
                model_selection.aselect_model,
                model_selection_llm=model_selection_llm,
                description_length=100
            )
        
        if settings.monitor_enabled:
            from autoschedule import ResourceMonitor
            import threading
            
            self._resource_monitor_client = ResourceMonitor(
                host_info_file_path=self._settings.host_info_file_path
            )
            # 守护线程
            threading.Thread(target=self._resource_monitor_client.start(), daemon=True).start()
            
        self._settings = settings
    
    def get_sql_session(self):
        """Provide database session
        """
        if not hasattr(self, '_mysql_server'):
            raise MySQLServerNotExistError("No available MySQL database server")
        
        session_generator = self._mysql_server.get_session_generator()
        try:
            session = next(session_generator)
            return session
        finally:
            session.close()
    
    def get_train_func(self, model_type: str) -> Callable:
        """Get the train func for the create_training_job()"""
        from autotrain import AutoTrainFunc
        
        train_func = AutoTrainFunc.from_model_type(model_type)
        return train_func
    
    async def select_models(
        self, 
        user_input: str, 
        task: str, 
        model_nums: int = 1
    ):
        try:
            models = await self.aselect(
                user_input=user_input,
                task=task,
                top_k=model_nums * 2,
                model_nums=model_nums
            )
        except Exception as e:
            raise SelectModelError(f"Failed to select the candidate model, for a specific reason: {e}")
        return models
    
    def create_training_job(
        self, 
        name: str, 
        func: Callable, 
        parameters: Optional[Dict[str, Any]] = None,
        host_ip: Optional[str] = None
    ):
        try:
            self._training_client.create_tfjob_from_func(
                name=name,
                func=func,
                parameters=parameters,
                base_image=self._settings.base_image,
                namespace=self._settings.namespcae,
                num_worker_replicas=1,
                host_ip=host_ip
            )
        except Exception as e:
            raise CreateJobError(f"Failed to create a training job '{name}', for a specific reason: {e}")
    
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
    
    def get_training_job_status(self, name: str):
        """Get the Training Job status."""
        try:
            status = self._training_client.get_tfjob(
                name=name,
                namespace=self._settings.namespcae
            ).status
            return status
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
        return self._resource_monitor_client.get_gpu_and_host(threshold=threshold)