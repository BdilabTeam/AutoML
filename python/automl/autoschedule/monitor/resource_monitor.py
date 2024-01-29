import asyncio
import json
import os.path

from .monitoring import GPUMonitor
from .scheduler_utils import get_gpu_index


class ResourceMonitor:
    def __init__(self, host_info_dir="./", host_info_file="host_info.json"):
        with open(os.path.join(host_info_dir, host_info_file), "r", encoding="utf-8") as f:
            host_config = json.load(f)
            host_info = host_config["host_info"]
        self._host_info = dict()
        for item in host_info:
            host_ip = item.pop('host_ip')
            # 将剩余的键值对添加到新字典中
            self._host_info[host_ip] = item

        self.gpu_monitor = GPUMonitor(self._host_info)

    async def start(self):
        while True:

            # 小优化 稍微自旋一下
            for i in range(30):
                self.gpu_monitor.update()

            # 2s 一次 让出时间给其他协程
            await asyncio.sleep(2)

    async def get_gpu_and_host(self, threshold):
        '''

        :param threshold: 该任务调用所要的GPU
        :return: None 如果所有Node的所有GPU都没有资源
        :return: host_ip,gpu_index 对应的host_ip与gpu_index
        '''
        host_ip, gpu_index = get_gpu_index(self._host_info, threshold)

        if gpu_index == -1:
            return None

        return host_ip, gpu_index


if __name__ == '__main__':
    tse = ResourceMonitor()
