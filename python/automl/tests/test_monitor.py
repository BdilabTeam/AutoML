import os
from autoschedule.monitor import ResourceMonitor
import pytest

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

class Monitor:
    @pytest.fixture
    def resource_monitor():
        rm = ResourceMonitor(
            host_info_file_path=os.path.join(PARENT_DIR, 'autoselect', 'host_info.json')
        )
        return rm
    
    def test_get_gpu_and_host(resource_monitor: ResourceMonitor):
        host_ip, gpu_index = resource_monitor.get_gpu_and_host(10)