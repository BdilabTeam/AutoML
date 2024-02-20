import os
from autoschedule.monitor import ResourceMonitor
import pytest

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

class Monitor:
    @pytest.fixture
    def resource_monitor(self):
        return ResourceMonitor(
            host_info_file_path=os.path.join(PARENT_DIR, 'autoselect', 'host_info.json')
        )
    
    def test_get_gpu_and_host(self, resource_monitor: ResourceMonitor):
        host_ip, gpu_index = resource_monitor.get_gpu_and_host(10)