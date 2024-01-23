from automl.monitor import ResourceMonitor
import pytest

class Monitor:
    @pytest.fixture
    def resource_monitor():
        rm = ResourceMonitor(
            host_info_dir="/Users/treasures_y/Documents/code/HG/AutoML/python/automl/tests",
            host_info_file="/Users/treasures_y/Documents/code/HG/AutoML/python/automl/tests/test_host_info.json"
        )
        return rm
    
    def test_get_gpu_and_host(resource_monitor: ResourceMonitor):
        host_ip, gpu_index = resource_monitor.get_gpu_and_host(10)