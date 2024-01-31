import json
import time

from .monitoring import GPUMonitor
from .scheduler_utils import get_gpu_index


class ResourceMonitor:
    def __init__(self, host_info_file_path: str):
        with open(host_info_file_path, "r", encoding="utf-8") as f:
            host_config = json.load(f)
            host_info = host_config["host_info"]
        self._host_info = dict()
        for item in host_info:
            host_ip = item.pop('host_ip')
            # 将剩余的键值对添加到新字典中
            self._host_info[host_ip] = item

        self.gpu_monitor = GPUMonitor(self._host_info)

    def start(self):
        while True:
            for i in range(30):
                self.gpu_monitor.update()
            time.sleep(1)


    def get_gpu_and_host(self, threshold):
        '''
        :param threshold: 该任务调用所要的GPU
        :return: None 如果所有Node的所有GPU都没有资源
        :return: host_ip,gpu_index 对应的host_ip与gpu_index
        '''
        host_ip, gpu_index = get_gpu_index(self._host_info, threshold)

        if gpu_index == -1:
            return None

        return host_ip, gpu_index