from automl.monitor import ResourceMonitor

def test_resource_monitor():
    
    rm = ResourceMonitor(
        host_info_dir="/Users/treasures_y/Documents/code/HG/AutoML/python/automl/tests",
        host_info_file="/Users/treasures_y/Documents/code/HG/AutoML/python/automl/tests/test_host_info.json"
    )
    host_ip, gpu_index = rm.get_gpu_and_host(10)

if __name__=="__main__":
    test_resource_monitor()