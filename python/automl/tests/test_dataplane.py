import pytest
from alserver.handlers.dataplane import DataPlane
from alserver import Settings

TASK_TYPE = 'structured-data-classification'
MODEL_TYPE = 'densenet'
TUNER = 'bayesian'
# 容器内数据卷路径
INPUTS = '/autotrain/autotrain/datasets/train.csv'

JOB_NAME = 'test-densenet'
HOST_IP = '60.204.186.96'


@pytest.mark.asyncio
class TestDataplane:
    @pytest.fixture(scope='class')
    def settings(self):
        return Settings(
            kubernetes_enabled=True,
            model_selection_enabled=True
        )
    @pytest.fixture(scope='class')
    def dataplane(self, settings):
        return DataPlane(settings=settings)

    def test_create_training_job(self, dataplane: DataPlane):
        train_func = dataplane.get_train_func(model_type=MODEL_TYPE)
        trainer_args_dict = {
            'task_type': TASK_TYPE,
            'model_type': MODEL_TYPE,
            'max_trials': 1,
            'tuner': TUNER,
            'inputs': INPUTS
        }
        dataplane.create_training_job(
            name=JOB_NAME,
            func=train_func,
            parameters=trainer_args_dict,
            host_ip=HOST_IP
        )
        
    def test_delete_training_job(self, dataplane: DataPlane):
        dataplane.delete_training_job(JOB_NAME)
    
    async def test_aselect_models(self, dataplane: DataPlane):
        models_info = await dataplane.aselect_models(
            user_input='I want a structured data classification model',
            task=TASK_TYPE,
            model_nums=1
        )
        print(models_info)
    
    def test_get_training_job_conditions(self, dataplane: DataPlane):
        conditions = dataplane.get_training_job_conditions(JOB_NAME)
        print(conditions)
        
    def test_get_training_job_status(self, dataplane: DataPlane):
        status = dataplane.get_training_job_status(JOB_NAME)
        print(status)
        print(status.start_time)
        print(status.completion_time)
        
    def test_is_training_job_succeeded(self, dataplane: DataPlane):
        res = dataplane.is_training_job_succeeded(name=JOB_NAME)
        assert res == True, 'Job completion status is False'
    
    def test_get_training_job_logs(self, dataplane: DataPlane):
        logs = dataplane.get_training_job_logs(JOB_NAME)
        print(logs)