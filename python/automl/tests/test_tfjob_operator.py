from automl.operators import KubeConfigSettings, TFJobSettings, TFJobOrchestrator
import pytest

KUBE_CONFIG_FILE=""

class TestOperator:
    @pytest.fixture
    def tfjob_orchestrator(self):
        kube_settings = KubeConfigSettings(kube_config_file=KUBE_CONFIG_FILE)
        return TFJobOrchestrator(settings=kube_settings)

    def test_construct_kubeflow_v1_tfjob(self, tfjob_orchestrator: TFJobOrchestrator):
        tfjob_settings = TFJobSettings()
        tfjob = tfjob_orchestrator.construct_kubeflow_v1_tfjob(settings=tfjob_settings)
        print(f"tfjob: {tfjob}")