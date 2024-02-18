import pytest
from alserver.handlers.dataplane import DataPlane
from alserver import Settings
from alserver.operators import TrainingClient

TASK_TYPE = 'structured-data-classification'
MODEL_TYPE = 'densenet'
TUNER = 'bayesian'
# 容器内数据卷路径
INPUTS = '/autotrain/autotrain/datasets/structured-data-classification.csv'

EXPERIMENT_JOB_NAME = 'test-7'
NAMESPACE = 'zauto'
HOST_IP = '60.204.186.96'

class TestTrainingClient:
    @pytest.fixture(scope='class')
    def training_client(self):
        settings = Settings(
            kubernetes_enabled=True,
            model_selection_enabled=True
        )
        dataplane = DataPlane(settings=settings)
        return dataplane._training_client
    
    def test_get_experiment_job_conditions(self, training_client: TrainingClient):
        conditions = training_client.get_job_conditions(
            name=EXPERIMENT_JOB_NAME,
            namespace=NAMESPACE
        )
        assert conditions is not None, 'Failed to get the job conditions'
        print(conditions)
        
    def test_get_training_experiment_job_status(self, training_client: TrainingClient):
        experiment_job_status = training_client.get_tfjob(
            name=EXPERIMENT_JOB_NAME, 
            namespace=NAMESPACE
        ).status
        assert experiment_job_status is not None, 'Failed to get the job status'
        print(experiment_job_status)
        print(experiment_job_status.start_time)
        print(experiment_job_status.completion_time)
        
    def test_is_training_job_succeeded(self, training_client: TrainingClient):
        res = training_client.is_job_succeeded(
            name=EXPERIMENT_JOB_NAME, 
            namespace=NAMESPACE
        )
        assert res == True, 'Job completion status is False'


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

    def test_create_experiment(self, dataplane: DataPlane):
        pass
    
    def test_delete_experiment_job(self, dataplane: DataPlane):
        dataplane.delete_experiment_job(EXPERIMENT_JOB_NAME)
    
    async def test_aselect_models(self, dataplane: DataPlane):
        models_info = await dataplane.aselect_models(
            user_input='I want a structured data classification model',
            task=TASK_TYPE,
            model_nums=1
        )
        print(models_info)