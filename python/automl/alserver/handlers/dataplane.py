import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Literal, List, TypeVar

from ..settings import Settings
from ..databases.mysql import MySQLClient
from ..errors import (
    MySQLNotExistError, 
    SelectModelError, 
    DeleteExperimentJobError,
    CreateExperimentJobError,
    GetExperimentJobLogsError,
    GetTrainingParamsError,
    SaveTrainingParamsError,
    ExperimentNotExistError,
    GetSessionError,
    ParseExperimentSummaryError,
    ValueError,
    GetExperimentJobStatusError,
    ExperimentNameError
)
from ..operators import TrainingClient
from ..utils import dataplane_utils, get_logger
from ..schemas import output_schema

UploadFile = TypeVar('UploadFile')

logger = get_logger(__name__)
from datetime import datetime
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
                model_type=str.lower(model.id),
                reason=model.reason,
                training_params=training_params_dict
            )
            candidate_models.append(candidate_model)
        return output_schema.CandidateModels(candidate_models=candidate_models)

    
    @transactional
    def create_experiment(
        self, 
        experiment_name: str,
        task_type: str,
        task_desc: str,
        model_type: str,
        training_params: Dict[str, Any],
        file_type: Literal['csv', 'image_folder'],
        files: List[UploadFile],
        host_ip: str,
        **kwargs
    ) -> output_schema.ExperimentInfo:
        from autotrain import AutoTrainFunc
        from ..models.experiment import Experiment
        from ..cruds.experiment import create_experiment, get_experiment
        from kubeflow.training.constants import constants
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        
        if get_experiment(session=session, experiment_name=experiment_name):
                raise ExperimentNameError("The experiment name has already been used, please re-enter it.")
            
        try:
            logger.info("Database - Add experiment items")
            experiment = Experiment(experiment_name=experiment_name, task_type=task_type, task_desc=task_desc, model_type=model_type)
            experiment = create_experiment(session=session, experiment=experiment)
            workspace_dir = dataplane_utils.generate_experiment_workspace_dir(experiment_name=experiment_name)
            experiment.workspace_dir = workspace_dir
            
            logger.info("Parse and store data")
            data_dir = os.path.join(workspace_dir, 'datasets')
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
                    
            logger.info("Get the training function and its parameters")
            train_func = AutoTrainFunc.from_model_type(model_type)
            training_params.update(
                {
                    'tp_project_name': dataplane_utils.TP_PROJECT_NAME,
                    'task_type': task_type,
                    'model_type': model_type,
                    'inputs': inputs,
                    'tp_directory': dataplane_utils.WORKSPACE_DIR_IN_CONTAINER
                }
            )
            tp_max_trials = kwargs.pop('tp_max_trials', None)
            if tp_max_trials:
                training_params['tp_max_trials'] = tp_max_trials
            tp_tuner = kwargs.pop("tp_tuner", None)
            if tp_tuner:
                training_params['tp_tuner'] = tp_tuner
            
            logger.info(f"Saving the training parameter.")
            training_params_file_path = dataplane_utils.get_experiment_training_params_file_path(experiment_name=experiment_name)
            try:
                dataplane_utils.save_dict_to_json_file(data=training_params, json_file=training_params_file_path)
            except Exception as e:
                raise SaveTrainingParamsError(f"Failed to save the training parameters, for a specific reason: {e}")
            
            logger.info("Publishing the experiment job.")
            try:
                self._training_client.create_tfjob_from_func(
                    name=experiment_name,
                    func=train_func,
                    parameters=training_params,
                    base_image=self._settings.base_image,
                    namespace=self._settings.namespcae,
                    num_worker_replicas=1,
                    host_ip=host_ip,
                    external_workspace_dir=workspace_dir,
                )
                self._training_client.wait_for_job_conditions(
                    name=experiment_name,
                    namespace=self._settings.namespcae,
                    expected_conditions=set([constants.JOB_CONDITION_CREATED]),
                    timeout=30,
                    polling_interval=1
                )
                
            except Exception as e:
                raise CreateExperimentJobError(f"Failed to create a experiment job '{experiment_name}', for a specific reason: {e}")
            
            return output_schema.ExperimentInfo(
                experiment_id=experiment.id,
                experiment_name=experiment.experiment_name,
                task_type=experiment.task_type,
                model_type=experiment.model_type,
            )
        except:
            logger.error("Failed to create, start rollback operation")
            dataplane_utils.remove_workspace_dir(workspace_dir=workspace_dir)
            self.delete_experiment_job(experiment_name=experiment_name)
            raise


    @transactional
    def delete_experiment(self, experiment_name: str, **kwargs):
        from ..cruds.experiment import delete_experiment, get_experiment
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        
        try: 
            experiment = get_experiment(session=session, experiment_name=experiment_name)
        except:    
            raise ExperimentNotExistError(f"Name: {experiment_name} for experiment does not exist.")
        
        self.delete_experiment_job(experiment_name=experiment_name)
        delete_experiment(session=session, experiment_name=experiment_name)
        dataplane_utils.remove_workspace_dir(workspace_dir=experiment.workspace_dir)


    def get_experiment_overview(self, experiment_name: str) -> output_schema.ExperimentOverview:
        from ..cruds.experiment import get_experiment
        from kubeflow.training.constants import constants
        
        session = self.get_session()
        try: 
            experiment = get_experiment(session=session, experiment_name=experiment_name)
        except:    
            raise ExperimentNotExistError(f"Name: {experiment_name} for experiment does not exist.")
        
        logger.info("Getting the status of the experiment job")
        try:
            experiment_job_status = self._training_client.get_tfjob(name=experiment_name, namespace=self._settings.namespcae).status
        except Exception as e:
            raise GetExperimentJobStatusError(f"Failed to get the status of the experiment '{experiment_name}', for a specific reason: {e}")
        
        if not experiment_job_status:
            raise ValueError("Experiment job status cannot be None")
        
        experiment_start_time = experiment_job_status.start_time.strftime("%Y-%m-%d %H:%M:%S") if experiment_job_status.start_time else None
        if experiment_job_status.completion_time:
            experiment_completion_time = experiment_job_status.completion_time.strftime("%Y-%m-%d %H:%M:%S") 
            experiment_duration_time = str(datetime.strptime(experiment_completion_time, "%Y-%m-%d %H:%M:%S") -  datetime.strptime(experiment_start_time, "%Y-%m-%d %H:%M:%S")).split(".")[0]
        else:
            experiment_completion_time = None
            experiment_duration_time = None
        conditions = experiment_job_status.conditions
        if conditions:
            for c in reversed(conditions):
                if c.status == constants.CONDITION_STATUS_TRUE:
                    experiment_status = c.type
                    break
        else:
            experiment_status = 'Unknown'
        
        if experiment_status == constants.JOB_CONDITION_SUCCEEDED:
            logger.info("Getting the summary of the experiment.")
            try:
                with open(dataplane_utils.get_experiment_summary_file_path(experiment_name=experiment_name)) as f:
                    summary = json.load(f)
            except Exception as e:
                raise ParseExperimentSummaryError(f"Failed to parse the summary of the experiment, for a specific reason: {e}")
            
            best_model_tracker = summary.get("best_model_tracker")
            if best_model_tracker:
                best_model = output_schema.BestModel(
                history=best_model_tracker.get("history"),
                parameters=best_model_tracker.get("hyperparameters").get("values") if best_model_tracker.get("hyperparameters") else None,
                model_graph_url=re.sub(r"metadata", os.path.join("metadata", experiment_name), best_model_tracker.get('model_graph_path')) if  best_model_tracker.get('model_graph_path') else ""
            )
            else:
                raise ValueError("Failed to get the 'best_model_tracker' key of the 'summary dict'")
            trials_tracker = summary.get('trials_tracker')
            if trials_tracker:
                trials = []
                for trial in trials_tracker.get("trials"):
                    trials.append(
                        output_schema.Trial(
                            trial_id=trial.get("trial_id"),
                            trial_status=trial.get("status"),
                            default_metric=round(trial.get("score"), 5),
                            best_step=trial.get('best_step'),
                            parameters=trial.get('hyperparameters').get('values') if trial.get('hyperparameters') else None,
                            model_graph_url=re.sub(r"metadata", os.path.join("metadata", experiment_name), trial.get('model_graph_path')) if trial.get('model_graph_path') else ""
                        )
                    )
            else:
                raise ValueError("Failed to get the 'trials_tracker' key of the 'summary dict'")
            experiment_summary_url = dataplane_utils.get_experiment_summary_file_url(experiment_name=experiment_name)
        else:
            logger.info("Experiment job is incomplete. Can't get the summary.")
            best_model = None
            trials = None
            experiment_summary_url = None
            
        return output_schema.ExperimentOverview(
            experiment_name=experiment.experiment_name,
            experiment_status=experiment_status,
            experiment_start_time=experiment_start_time,
            experiment_completion_time=experiment_completion_time,
            experiment_duration_time=experiment_duration_time,
            experiment_summary_url=experiment_summary_url,
            tuner=None,
            trials=trials,
            best_model=best_model
        )

    
    @transactional
    def get_experiment_cards(self, **kwargs) -> output_schema.ExperimentCards:
        from ..cruds.experiment import get_all_experiments
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        
        experiments = get_all_experiments(session=session)
        experiment_cards = []
        for experiment in experiments:
            experiment_card = output_schema.ExperimentCard(
                experiment_name=experiment.experiment_name,
                task_type=experiment.task_type,
                task_desc=experiment.task_desc,
                model_type=experiment.model_type,
                experiment_job_name=experiment.experiment_job_name
            )
            experiment_cards.append(experiment_card)
        return output_schema.ExperimentCards(experiment_cards=experiment_cards)
    
    def delete_experiment_job(self, experiment_name: str):
        try:
            self._training_client.delete_tfjob(
                name=experiment_name, 
                namespace=self._settings.namespcae
            )
        except Exception as e:
            raise DeleteExperimentJobError(f"Failed to delete a experiment job '{experiment_name}', for a specific reason: {e}")

    
    async def get_experiment_logs(self, experiment_name: str, websocket = None):
        try:
            await self._training_client.get_job_logs(
                name=experiment_name,
                namespace=self._settings.namespcae,
                is_master=False,
                replica_type='worker',
                follow=True,
                websocket=websocket
            )   
        except Exception as e:
            await websocket.close(reason="Log acquisition process exception.")
            raise GetExperimentJobLogsError(f"Failed to get the logs of the experiment job '{experiment_name}'")

    
    def get_gpu_and_host(self, threshold):
        return self._resource_monitor_service.get_gpu_and_host(threshold=threshold)
    
    def get_model_repository(self) -> output_schema.ModelRepository:
        from ..cruds.experiment import get_all_experiments
        
        session = self.get_session()
        experiments = get_all_experiments(session=session)
        models = []
        for experiment in experiments:
            if self._training_client.is_job_succeeded(name=experiment.experiment_name, namespace=self._settings.namespcae):
                models.append(experiment.experiment_name)
        return output_schema.ModelRepository(models=models)
        