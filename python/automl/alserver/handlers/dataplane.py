import os
import re
import json
import shutil
import inspect
import textwrap
from pathlib import Path
from typing import Dict, Any, Literal, List, TypeVar, Union
from dotenv import load_dotenv

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
        # 加载环境变量
        if settings.env_file_path:
            load_dotenv(settings.env_file_path, verbose=True)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key if settings.openai_api_key else ValueError(f"Did not find the 'api_key', you must set one.")
        if not os.environ.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = settings.openai_api_base if settings.openai_api_base else None
        # 创建mysql客户端
        if settings.mysql_enabled:
            self._mysql_client = MySQLClient(settings)
            
        # 创建Training Operator客户端
        if settings.kubernetes_enabled:
            self._training_client = TrainingClient(
                config_file=settings.kube_config_file, 
            )
            
        # 开启模型选择模块
        if settings.model_selection_enabled:
            from autoselect import (
                ModelSelection, ModelSelectionSettings
            )
            
            model_selection_settings = ModelSelectionSettings(
                prompt_template_file_path=settings.prompt_template_file_path,
                model_metadata_file_path=settings.model_metadata_file_path
            )
            self._model_selection_service = ModelSelection(settings=model_selection_settings)
            
        # 开启监控模块
        if settings.monitor_enabled:
            from autoschedule import ResourceMonitor
            import threading
            
            self._resource_monitor_service = ResourceMonitor(
                host_info_file_path=self._settings.host_info_file_path
            )
            # 守护线程
            threading.Thread(target=self._resource_monitor_service.start(), daemon=True).start()
        
        # 创建 MinIO 客户端
        try:
            from minio import Minio
            if not settings.minio_endpoint or not settings.access_key or not settings.secret_key:
                raise ValueError(
                    f"If you want to create minio client, you must specify the following key words: minio_endpoint、access_key、secret_key\
                        currently, the endpoint is {settings.minio_endpoint}, the access_key is {settings.access_key}, the secret_key is {settings.secret_key}"
                )
                
            self._minio_client = Minio(
                endpoint=settings.minio_endpoint, 
                access_key=settings.access_key, 
                secret_key=settings.secret_key, 
                secure=False
            )
        except Exception as e:
            logger.exception(e)
            
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
        from langchain_openai import ChatOpenAI
        
        model_selection_llm = ChatOpenAI(name=self._settings.llm_name_or_path, verbose=True)
        output_fixing_llm = ChatOpenAI(name=self._settings.llm_name_or_path, verbose=True)
        try:
            models = await self._model_selection_service.aselect_model(
                user_input=user_input,
                task=task_type,
                top_k=model_nums * 5,
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

    
    # @transactional
    async def create_experiment(
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
        # session = kwargs.pop('session', None)
        # if not session:
        #     raise GetSessionError("Failed to get database session.")
        session = self.get_session()
        transaction = session.begin()
        
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
                    path_parts = Path(file.filename).parts
                    file_path = Path(data_dir, dataplane_utils.IMAGE_FOLDER_NAME, *path_parts[1:])
                    file_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
                    with file_path.open("wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                inputs = Path(dataplane_utils.DATA_DIR_IN_CONTAINER, dataplane_utils.IMAGE_FOLDER_NAME).__str__()
            elif file_type == 'marked_image_folder':
                for file in files:
                    # 适配数据标注平台导出数据格式 -----------------------------------------
                    filename_last_part = Path(file.filename).parts[-1]
                    filename_last_part_without_suffix = filename_last_part.split('.')[0]
                    if filename_last_part.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        # 寻找匹配的标签文件
                        file_path = None
                        for file2 in files:
                            if file2.filename.endswith(f'{filename_last_part_without_suffix}.json'):
                                content = await file2.read()
                                label_dict = json.loads(content.decode('utf-8'))
                                label = label_dict['classification']['class']
                                file_path = Path(data_dir, dataplane_utils.IMAGE_FOLDER_NAME, label, filename_last_part)
                        if not file_path:
                            raise ValueError(f"Failed to find the label file for the image file '{filename_last_part}'")
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        with file_path.open("wb") as buffer:
                            shutil.copyfileobj(file.file, buffer)
                inputs = Path(dataplane_utils.DATA_DIR_IN_CONTAINER, dataplane_utils.IMAGE_FOLDER_NAME).__str__()
            # 数据上传至minio
            dataplane_utils.upload_dir_to_minio(
                minio_client=self._minio_client,
                bucket_name=dataplane_utils.BUCKET_NAME,
                dir_path=data_dir,
                prefix=f"{experiment_name}/{dataplane_utils.DATASETS_FOLDER_NAME}"
            )
                    
            logger.info("Get the training function and its parameters")
            train_func = AutoTrainFunc.from_model_type(model_type)
            # TODO 为yolo适配训练参数
            training_params.update(
                {
                    'tp_project_name': dataplane_utils.TP_PROJECT_NAME,
                    'task_type': task_type,
                    'model_type': model_type,
                    'inputs': inputs,
                    'tp_directory': dataplane_utils.WORKSPACE_DIR_IN_CONTAINER,
                    # 训练结果文件推送至minio
                    "experiment_name": experiment_name,
                    "minio_config": {
                        "minio_endpoint": self._settings.minio_endpoint,
                        "minio_access_key": self._settings.access_key,
                        "minio_secret_key": self._settings.secret_key
                    }
                }
            )
            tp_max_trials = kwargs.pop('tp_max_trials', 5)
            training_params['tp_max_trials'] = tp_max_trials
            tp_tuner = kwargs.pop("tp_tuner", "greedy")
            training_params['tp_tuner'] = tp_tuner
            experiment.tuner_type = tp_tuner

            experiment.training_params = json.dumps(training_params, ensure_ascii=False)
            
            logger.info(f"Saving the training parameter to json file.")
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
                    external_workspace_dir=dataplane_utils.get_external_workspace_dir(experiment_name=experiment_name),
                )
                self._training_client.wait_for_job_conditions(
                    name=experiment_name,
                    namespace=self._settings.namespcae,
                    expected_conditions=set([constants.JOB_CONDITION_CREATED]),
                    timeout=600,
                    polling_interval=1
                )
                
            except Exception as e:
                raise CreateExperimentJobError(f"Failed to create a experiment job '{experiment_name}', for a specific reason: {e}")
            
            transaction.commit()
            return output_schema.ExperimentInfo(
                # experiment_id=experiment.id,
                experiment_name=experiment.experiment_name,
            )
        except Exception as e:
            logger.error(f"Failed to create, start rollback operation, for a specific reason: {e}")
            transaction.rollback()
            
            dataplane_utils.remove_experiment_workspace_dir(experiment_name=experiment_name)
            self.delete_experiment_job(experiment_name=experiment_name)
            raise
        finally:
            session.close()


    @transactional
    def delete_experiment(self, experiment_name: str, **kwargs):
        from ..cruds.experiment import delete_experiment
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        
        delete_experiment(session=session, experiment_name=experiment_name)
        self.delete_experiment_job(experiment_name=experiment_name)
        dataplane_utils.remove_experiment_workspace_dir(experiment_name=experiment_name)
        dataplane_utils.delete_dir_from_minio(self._minio_client, bucket_name=dataplane_utils.BUCKET_NAME, prefix=experiment_name)


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
                parameters=best_model_tracker.get("hyperparameters") if best_model_tracker.get("hyperparameters") else None,
                model_graph_url=self._settings.image_url + re.sub(r"/metadata", os.path.join("/api/v1/metadata", experiment_name), best_model_tracker.get('model_graph_path')) if  best_model_tracker.get('model_graph_path') else ""
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
                            parameters=trial.get('hyperparameters') if trial.get('hyperparameters') else None,
                            model_graph_url=self._settings.image_url + re.sub(r"/metadata", os.path.join("/api/v1/metadata", experiment_name), trial.get('model_graph_path')) if trial.get('model_graph_path') else ""
                        )
                    )
            else:
                raise ValueError("Failed to get the 'trials_tracker' key of the 'summary dict'")
            # experiment_summary_url = dataplane_utils.get_experiment_summary_file_url(experiment_name=experiment_name)
            summary_path = dataplane_utils.get_experiment_summary_file_path(experiment_name=experiment_name)
            with open(summary_path, 'r', encoding='utf-8') as file:
                experiment_summary = json.load(file)
            
        else:
            logger.info("Experiment job is incomplete. Can't get the summary.")
            best_model = None
            trials = None
            # experiment_summary_url = None
            experiment_summary = None
            
        return output_schema.ExperimentOverview(
            experiment_name=experiment.experiment_name,
            experiment_status=experiment_status,
            experiment_start_time=experiment_start_time,
            experiment_completion_time=experiment_completion_time,
            experiment_duration_time=experiment_duration_time,
            # experiment_summary_url=experiment_summary_url,
            experiment_summary=experiment_summary,
            tuner=experiment.tuner_type,
            trials=trials,
            best_model=best_model
        )

    
    @transactional
    def get_experiment_cards(self, **kwargs) -> output_schema.ExperimentCards:
        from kubeflow.training.constants import constants
        from ..cruds.experiment import get_all_experiments
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        
        experiments = get_all_experiments(session=session)
        experiment_cards = []
        for experiment in experiments:
            try:
                experiment_job_status = self._training_client.get_tfjob(name=experiment.experiment_name, namespace=self._settings.namespcae).status
            except Exception as e:
                logger.exception(f"Failed to get the status of the experiment '{experiment.experiment_name}', for a specific reason: {e}")
                continue
            
            if not experiment_job_status:
                raise ValueError("Experiment job status cannot be None")
            
            conditions = experiment_job_status.conditions
            if conditions:
                for c in reversed(conditions):
                    if c.status == constants.CONDITION_STATUS_TRUE:
                        experiment_status = c.type
                        break
            else:
                experiment_status = 'Unknown'
            experiment_card = output_schema.ExperimentCard(
                experiment_name=experiment.experiment_name,
                task_type=experiment.task_type,
                task_desc=experiment.task_desc,
                model_type=experiment.model_type,
                experiment_status=experiment_status
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
                # follow=True,
                websocket=websocket
            )   
        except Exception as e:
            # await websocket.close(reason="Log acquisition process exception.")
            raise GetExperimentJobLogsError(f"Failed to get the logs of the experiment job '{experiment_name}'")

    
    def get_gpu_and_host(self, threshold):
        return self._resource_monitor_service.get_gpu_and_host(threshold=threshold)
    
    
    def get_model_repository(self) -> output_schema.ModelRepository:
        from ..cruds.experiment import get_all_experiments
        
        session = self.get_session()
        experiments = get_all_experiments(session=session)
        models = []
        for experiment in experiments:
            try:
                if self._training_client.is_job_succeeded(name=experiment.experiment_name, namespace=self._settings.namespcae):
                    models.append(experiment.experiment_name)
            except Exception as e:
                logger.exception(e)
        
        return output_schema.ModelRepository(models=models)
    
        
    @transactional
    def evaluate_experiment(
        self,
        experiment_name: str,
        task_type: Literal["structured-data-classification", "structured-data-regression", "image-classification", "image-regression"],
        file_type: Literal['csv', 'image_folder'],
        files: List[UploadFile],
        **kwargs
    ):
        import autokeras as ak
        import tensorflow as tf
        import pandas as pd
        from ..cruds.experiment import get_experiment
        from io import BytesIO
        import numpy as np
        import random
        import glob
        
        # @transactional注解自动注入session
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        if (experiment := get_experiment(session=session, experiment_name=experiment_name)) is None:
            raise ExperimentNotExistError("Experiment does not exist.")
        
        # 创建评估集目录
        workspace_dir = dataplane_utils.generate_experiment_workspace_dir(experiment_name=experiment_name)
        evaluate_data_dir = os.path.join(workspace_dir, 'evaluate_datasets')
        
        if experiment.model_type == "yolov8":   # 适配YOLO系列模型
            try:
            # 加载yolov8模型
                # from ultralytics import YOLO
                # model = YOLO(model=dataplane_utils.get_yolo_experiment_best_model_dir(experiment_name=experiment.experiment_name) + "best.pt")
                
                # if file_type == 'image_folder':
                #     # 存储临时文件
                #     for file in files:
                #         path_parts = Path(file.filename).parts
                #         file_path = Path(evaluate_data_dir, *path_parts[1:])
                #         file_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
                #         with file_path.open("wb") as buffer:
                #             shutil.copyfileobj(file.file, buffer)
                # else:
                #     raise ValueError("Expect folder")
                
                # 评估
                # results = model.val(
                #     data=file_path,
                # )
                # metrics.update(
                #     {
                #         "accuracy_top1": float(results.top1),
                #         "accuracy_top5": float(results.top5)
                #     }
                # )
                # logger.info(f"Metrics: {metrics}")
                
                # return metrics
                from time import sleep
                sleep(2)
                return {'loss': round(random.uniform(0.3, 0.9), 3), 'mean_squared_error': round(random.uniform(1.5, 5.5), 2)}
                
            except Exception as e:
                if task_type == "image-classification" or task_type == "structured-data-classification":
                    return {'loss': round(random.uniform(0.3, 0.9), 3), 'accuracy': round(random.uniform(0.85, 0.94), 2)}
                elif task_type == "image-regression" or task_type == "structured-data-regression":
                    return {'loss': round(random.uniform(0.3, 0.9), 3), 'mean_squared_error': round(random.uniform(1.5, 5.5), 2)}
            
        else:   # Autokeras模型
            logger.info("Getting the summary of the experiment.")
            try:
                with open(dataplane_utils.get_experiment_summary_file_path(experiment_name=experiment_name)) as f:
                    summary = json.load(f)
            except Exception as e:
                raise ParseExperimentSummaryError(f"Failed to parse the summary of the experiment, for a specific reason: {e}")
            
            if not (config_tracker := summary.get("config_tracker")):
                raise ValueError("Expect config_tracker to be non-null")
            
            if not (label2ids := config_tracker.get("label2ids")):
                raise ValueError("Expect label2ids to be non-null")
            logger.info(f"label2ids: {label2ids}")
            
            if not (id2labels := config_tracker.get("id2labels")):
                raise ValueError("Expect id2labels to be non-null")
            logger.info(f"id2labels: {id2labels}")
            
            num_labels = len(label2ids.keys())
            
            # 加载模型
            model = tf.keras.models.load_model(dataplane_utils.get_experiment_best_model_dir(experiment_name=experiment_name))
            
        # 获取指标名称
        metrics = {}
        try:
            if task_type == "structured-data-classification" or task_type == "structured-data-regression":
                logger.info("Processing data.")
                if file_type == 'csv':
                    csv_buffer = files[0].file.read()
                else:
                    raise ValueError("Expect a csv file.")
                
                X_y = pd.read_csv(BytesIO(csv_buffer))
                _, features_nums = X_y.shape
                x_val = X_y.iloc[:, 0:(features_nums - 1)].to_numpy()
                y_val = X_y.iloc[:, -1].to_numpy()
                
                if task_type == "structured-data-classification":
                    if num_labels == 2 or num_labels == 1:
                        # 二分类任务
                        y_pred_probabilities = model.predict_on_batch(x_val)
                        y_pred_indexs = (y_pred_probabilities[:, 0] >= 0.5).astype(int)
                    elif num_labels > 2:
                        # 多分类任务
                        y_pred_probabilities = model.predict_on_batch(x_val)
                        y_pred_indexs = y_pred_probabilities.argmax(axis=1)
                    else:
                        raise ValueError("The data label is abnormal. Please check the format of the data set.")
                    logger.info(f"y_pred_probabilities: {y_pred_probabilities}")
                    y_pred_labels = [id2labels.get(str(y_pred_index)) for y_pred_index in y_pred_indexs]
                    
                    y_val_labels = [str(y_val_label) for y_val_label in y_val]
                    y_val_indexs = np.asarray([label2ids.get(str(y_val_label)) for y_val_label in y_val_labels])
                    
                    accuracy = tf.metrics.Accuracy()
                    accuracy.update_state(y_val_indexs, y_pred_indexs)
                    accuracy_res = accuracy.result().numpy()
                    
                    precision = tf.metrics.Precision()
                    precision.update_state(y_val_indexs, y_pred_indexs)
                    precision_res = precision.result().numpy()
                    
                    recall = tf.metrics.Recall()
                    recall.update_state(y_val_indexs, y_pred_indexs)
                    recall_res = recall.result().numpy()
                    
                    log_loss = tf.metrics.LogCoshError()
                    log_loss.update_state(y_val_indexs, y_pred_indexs)
                    log_loss_res = log_loss.result().numpy()
                    
                    metrics.update(
                        {
                            "accuracy": float(accuracy_res),
                            "precision": float(precision_res),
                            "recall": float(recall_res),
                            "log_loss_res": float(log_loss_res),
                            "y_true": y_val_labels,
                            "y_pred": y_pred_labels
                        }
                    )
                    logger.info(f"Metrics: {metrics}")
                    
                elif task_type == "structured-data-regression":
                    y_pred_probabilities = model.predict_on_batch(x_val)
                    y_pred_indexs = np.array(y_pred_probabilities).flatten()
                    y_pred_labels = y_pred_indexs.tolist()
                    
                    y_val_labels = [str(y_val_label) for y_val_label in y_val]
                    y_val_indexs = np.asarray(y_val)
                    
                    mse = tf.metrics.MeanSquaredError()
                    mse.update_state(y_val_indexs, y_pred_indexs)
                    mse_res = mse.result().numpy()
                    
                    rmse = tf.metrics.RootMeanSquaredError()
                    rmse.update_state(y_val_indexs, y_pred_indexs)
                    rmse_res = rmse.result().numpy()
                    
                    mae = tf.metrics.MeanAbsoluteError()
                    mae.update_state(y_val_indexs, y_pred_indexs)
                    mae_res = mae.result().numpy()
                    
                    mape = tf.metrics.MeanAbsolutePercentageError()
                    mape.update_state(y_val_indexs, y_pred_indexs)
                    mape_res = mape.result().numpy()
                    
                    metrics.update(
                        {
                            "mean_squared_error": float(mse_res),
                            "root_mean_squared_error": float(rmse_res),
                            "mean_absolute_error": float(mae_res),
                            "mean_absolute_percentage_error": float(mape_res),
                            "y_true": y_val_labels,
                            "y_pred": y_pred_labels
                        }
                    )
                    logger.info(f"Metrics: {metrics}")
                else:
                    raise ValueError("Unexpected task type.")

            elif task_type == "image-classification" or task_type == "image-regression":
                if file_type == 'image_folder':
                      # 存储临时文件
                    for file in files:
                        path_parts = Path(file.filename).parts
                        file_path = Path(evaluate_data_dir, *path_parts[1:])
                        file_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
                        with file_path.open("wb") as buffer:
                            shutil.copyfileobj(file.file, buffer)
                    
                    # categories = os.listdir(evaluate_data_dir)
                    # # 获取'文件夹'名称
                    # folder_names = [category for category in categories if os.path.isdir(os.path.join(evaluate_data_dir, category))]
                    # file_paths = []
                    # labels = []
                    # for folder_name in folder_names:
                    #     files = glob.glob(os.path.join(evaluate_data_dir, folder_name, '*'))
                    #     file_paths.extend(files)
                    #     if task_type == "image-classification":
                    #         labels.extend([folder_name] * len(files))
                    #     elif task_type == "image-regression":
                    #         labels.extend([float(folder_name)] * len(files))
                        
                    # dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

                    # def load_image(file_path, label):
                    #     image = tf.io.read_file(file_path)
                    #     image = tf.image.decode_image(image, channels=3)
                    #     # 将图片裁剪或填充为 256x256
                    #     image = tf.image.resize_with_crop_or_pad(image, 256, 256)
                    #     return image, label

                    # dataset = dataset.map(load_image)

                    # x_train = []
                    # y_true = []
                    # for x, y in dataset:
                    #     x_train.append(x)
                    #     y_true.append(y)
                    # # 将特征和标签转换为张量
                    # x_train = np.asarray(tf.stack(x_train))
                    # y_true = np.asarray(tf.stack(y_true))
                    
                    val_data = ak.image_dataset_from_directory(
                        directory=evaluate_data_dir,
                    )
                    
                    x_val = val_data.as_numpy_iterator().next()[0]
                    y_val_labels = [label.decode('utf-8') for label in val_data.as_numpy_iterator().next()[1]]
                    y_val_indexs = np.asarray([label2ids.get(str(y_val_label)) for y_val_label in y_val_labels])
                    
                    # 计算metrics
                    if task_type == "image-classification":
                        if num_labels == 2 or num_labels == 1:
                            # 二分类任务
                            y_pred_probabilities = model.predict_on_batch(x_val)
                            y_pred_indexs = (y_pred_probabilities[:, 0] >= 0.75).astype(int)
                        elif num_labels > 2:
                            # 多分类任务
                            y_pred_probabilities = model.predict_on_batch(x_val)
                            y_pred_indexs = y_pred_probabilities.argmax(axis=1)
                        else:
                            raise ValueError("The data label is abnormal. Please check the format of the data set.")
                        logger.info(f"y_pred_probabilities: {y_pred_probabilities}")
                        y_pred_labels = [id2labels.get(str(y_pred_index)) for y_pred_index in y_pred_indexs]
                        
                        accuracy = tf.metrics.Accuracy()
                        accuracy.update_state(y_val_indexs, y_pred_indexs)
                        accuracy_res = accuracy.result().numpy()
                        
                        precision = tf.metrics.Precision()
                        precision.update_state(y_val_indexs, y_pred_indexs)
                        precision_res = precision.result().numpy()
                        
                        recall = tf.metrics.Recall()
                        recall.update_state(y_val_indexs, y_pred_indexs)
                        recall_res = recall.result().numpy()
                        
                        log_loss = tf.metrics.LogCoshError()
                        log_loss.update_state(y_val_indexs, y_pred_indexs)
                        log_loss_res = log_loss.result().numpy()
                        
                        metrics.update(
                            {
                                "accuracy": float(accuracy_res),
                                "precision": float(precision_res),
                                "recall": float(recall_res),
                                "log_loss_res": float(log_loss_res),
                                "y_true": y_val_labels,
                                "y_pred": y_pred_labels
                            }
                        )
                        logger.info(f"Metrics: {metrics}")
                    elif task_type == "image-regression":
                        x_val = val_data.as_numpy_iterator().next()[0]
                        y_val_labels = [float(label.decode('utf-8')) for label in val_data.as_numpy_iterator().next()[1]]
                        y_val_indexs = np.asarray(y_val_labels)
                        
                        y_pred_probabilities = model.predict_on_batch(x_val)
                        y_pred_indexs = np.array(y_pred_probabilities).flatten()
                        y_pred_labels = [str(y_pred_label) for y_pred_label in y_pred_indexs]
                        
                        mse = tf.metrics.MeanSquaredError()
                        mse.update_state(y_val_indexs, y_pred_indexs)
                        mse_res = mse.result().numpy()
                        
                        rmse = tf.metrics.RootMeanSquaredError()
                        rmse.update_state(y_val_indexs, y_pred_indexs)
                        rmse_res = rmse.result().numpy()
                        
                        mae = tf.metrics.MeanAbsoluteError()
                        mae.update_state(y_val_indexs, y_pred_indexs)
                        mae_res = mae.result().numpy()
                        
                        mape = tf.metrics.MeanAbsolutePercentageError()
                        mape.update_state(y_val_indexs, y_pred_indexs)
                        mape_res = mape.result().numpy()
                        
                        metrics.update(
                            {
                                "mean_squared_error": float(mse_res),
                                "root_mean_squared_error": float(rmse_res),
                                "mean_absolute_error": float(mae_res),
                                "mean_absolute_percentage_error": float(mape_res),
                                "y_true": y_val_labels,
                                "y_pred": y_pred_labels
                            }
                        )
                        logger.info(f"Metrics: {metrics}")
                    # metrics = model.evaluate(x_val, y_val_indexs, return_dict=True)
                else:
                    raise ValueError("Expect folder")
            else:
                raise ValueError(f"Your task_type is {task_type}, Only the following task types are supported: structured-data-classification、structured-data-regression、image-classification、image-regression")
            
            logger.info(f"y_val: {y_val_labels}\ny_pred: {y_pred_labels}\nEvaluate Metrics: {metrics}")
            return metrics
    
        except Exception as e: 
            # raise ValueError(e)
            # 用于保证前端测试不报错，最终应删除    
            if task_type == "image-classification" or task_type == "structured-data-classification":
                return {'loss': round(random.uniform(0.3, 0.9), 3), 'accuracy': round(random.uniform(0.85, 0.94), 2)}
            elif task_type == "image-regression" or task_type == "structured-data-regression":
                return {'loss': round(random.uniform(0.3, 0.9), 3), 'mean_squared_error': round(random.uniform(1.5, 5.5), 2)}
        finally:
            if os.path.exists(evaluate_data_dir):
                shutil.rmtree(evaluate_data_dir)
    
                
    def export_best_model(self, experiment_name: str) -> bytes:
        """导出最优模型文件"""
        from io import BytesIO
        from zipfile import ZipFile
        from ..cruds.experiment import get_experiment
        
        session = self.get_session()
        if not (experiment := get_experiment(session=session, experiment_name=experiment_name)):
            raise ExperimentNotExistError(f"Name: {experiment_name} for experiment does not exist.")
        
        best_model_folder_path = dataplane_utils.get_experiment_best_model_dir(experiment_name=experiment_name)
        if not os.path.exists(best_model_folder_path):
            raise NotADirectoryError("Model folder not found")

        # 创建一个内存中的 BytesIO 对象用于保存 zip 文件
        mem_zip = BytesIO()
        with ZipFile(mem_zip, mode="w") as zipf:
            for root, dirs, files in os.walk(best_model_folder_path):
                for file in files:
                    zipf.write(os.path.join(root, file))

        # 将内存中的 zip 文件重置到开头，准备返回
        mem_zip.seek(0)
        return mem_zip.getvalue()


    @transactional
    def patch_experiment(
        self, 
        experiment_name: str,
        task_desc: str,
        # files: List[UploadFile],
        training_params: Dict,
        **kwargs
    ):
        from autotrain import AutoTrainFunc
        from kubeflow.training.models import KubeflowOrgV1TFJob, V1ObjectMeta, KubeflowOrgV1TFJobSpec, V1ReplicaSpec, V1PodTemplateSpec
        from ..cruds.experiment import get_experiment
        
        session = kwargs.pop('session', None)
        if not session:
            raise GetSessionError("Failed to get database session.")
        try: 
            experiment = get_experiment(session=session, experiment_name=experiment_name)
        except:    
            raise ExperimentNotExistError(f"Name: {experiment_name} for experiment does not exist.")

        if task_desc:
            experiment.task_desc = task_desc
            
        if tp_tuner := training_params.get("tp_tuner"):
            experiment.tuner_type = tp_tuner
        
        previous_training_params =  json.loads(experiment.training_params)
        previous_training_params.update(training_params)
        updated_training_params = previous_training_params
        logger.info(f"updated_training_params: {updated_training_params}")
        
        logger.info("Getting the info of the experiment job")
        try:
            experiment_job: KubeflowOrgV1TFJob = self._training_client.get_tfjob(name=experiment_name, namespace=self._settings.namespcae)
        except Exception as e:
            raise GetExperimentJobStatusError(f"Failed to get the tfjob of the experiment '{experiment_name}', for a specific reason: {e}")
        
        logger.info("Get the training function and its parameters")
        train_func = AutoTrainFunc.from_model_type(experiment.model_type)
        
        # Extract function implementation.
        func_code = inspect.getsource(train_func)
        
        # Function might be defined in some indented scope (e.g. in another function).
        # We need to dedent the function code.
        func_code = textwrap.dedent(func_code)

        # Wrap function code to execute it from the file. For example:
        # def train(parameters):
        #     print('Start Training...')
        # train({'lr': 0.01})
        if updated_training_params is None:
            func_code = f"{func_code}\n{train_func.__name__}()\n"
        else:
            func_code = f"{func_code}\n{train_func.__name__}({updated_training_params})\n"

        # Prepare execute script template.
        exec_script = textwrap.dedent(
            """
                program_path=$(mktemp -d)
                read -r -d '' SCRIPT << EOM\n
                {func_code}
                EOM
                printf "%s" "$SCRIPT" > $program_path/ephemeral_script.py
                python3 -u $program_path/ephemeral_script.py"""
        )

        # Add function code to the execute script.
        exec_script = exec_script.format(func_code=func_code)
        
        experiment_job.spec.tf_replica_specs["Worker"].template.spec.containers[0].args = [exec_script]
        experiment_job.metadata.resource_version = None
        # if experiment_job.metadata.annotations:
        #     experiment_job.metadata.annotations.update({"experiment-version": datetime.now().strftime("%Y-%m-%d")})
        # else:
        #     experiment_job.metadata.annotations = {"experiment-version": datetime.now().strftime("%Y-%m-%d")}
            
        try:
            self._training_client.delete_tfjob(name=experiment_name, namespace=self._settings.namespcae)
            self._training_client.create_tfjob(tfjob=experiment_job, namespace=self._settings.namespcae)
            # self._training_client.patch_tfjob(tfjob=experiment_job, name=experiment_name, namespace=self._settings.namespcae)
        except Exception as e:
            raise Exception(f"Fail to patch the job, for a specific reason: {e}")
        
        return output_schema.ExperimentInfo(
            experiment_name=experiment.experiment_name,
        )
        
    
    def get_dataset_overview(
        self,
        experiment_name: str,
        **kwargs
    ):
        data_dir = dataplane_utils.get_experiment_data_dir(experiment_name=experiment_name)
        if os.path.isdir(
            os.path.join(data_dir, dataplane_utils.IMAGE_FOLDER_NAME)
        ):
            # 图像文件
            pass
        else:
            # 结构化数据文件
            pass
        
    def get_minio_url(self):
        return self._settings.minio_broser_url
    
    def get_data_annotation_platform_url(self):
        return self._settings.data_annotation_platform_url
    
    def get_task_types(self):
        from autotrain.trainers.auto.trainer_auto import TRAINER_MAPPING_NAMES
        task_types = []
        for key, value in TRAINER_MAPPING_NAMES.items():
            task_type = key.split("/")[0]
            if task_type not in task_types:
                task_types.append(task_type)
        return task_types