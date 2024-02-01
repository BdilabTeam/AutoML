import os
from autoschedule.monitor import ResourceMonitor
import pytest

class Monitor:
    @pytest.fixture
    def resource_monitor():
        rm = ResourceMonitor(
            host_info_dir=os.path.dirname(os.path.abspath(__file__)),
            host_info_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_host_info.json')
        )
        return rm
    
    def test_get_gpu_and_host(resource_monitor: ResourceMonitor):
        host_ip, gpu_index = resource_monitor.get_gpu_and_host(10)