from fabric import Connection
from fabric.runners import Result
from fabric import SerialGroup as Group
from fabric.group import GroupResult
import re
import json
from datetime import datetime
import time

# from fabric import ThreadingGroup as Group

host_info = dict()
host_info['10.170.23.190'] = {'user': 'root', 'password': '123456', 'node_name': 'master'}

class GPUMonitor:
    def connect(self):
        self.connectons = list()
        for host in self.host_info:
            conn = Connection(host, user=self.host_info[host]['user'], port=22,
                              connect_kwargs={"password": self.host_info[host]['password']},
                              connect_timeout=200)
            conn.run('hostname')  # 这行不能忽略
            self.connectons.append(conn)

        self.prod = Group.from_connections(self.connectons)
        print(self.prod)
        return 0

    def total_gpu_memory_set(self):
        total_memory: GroupResult = self.prod.run('nvidia-smi --query-gpu=memory.total --format=csv', hide=True)
        for conn, result in total_memory.items():
            temp = []
            # 用换行符进行分割的原因是每行代表了一个GPU的信息
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                # 为什么是 :-4? 举例:"15360 MiB"
                temp.append(float(res[i][:-4]))
            self.host_info[conn.host]['gpu_memory_total'] = temp

    def gpu_utils_update(self):
        gpu_results: GroupResult = self.prod.run('nvidia-smi --query-gpu=utilization.gpu --format=csv', hide=True)
        # 分别进行字符串解析
        for conn, result in gpu_results.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                # 35 %
                temp.append(float(res[i][:-2]))
            self.host_info[conn.host]['gpu_utils'] = temp

    def gpu_memory_used_update(self):
        used_memory: GroupResult = self.prod.run('nvidia-smi --query-gpu=memory.used --format=csv', hide=True)
        for conn, result in used_memory.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                temp.append(float(res[i][:-4]))
            self.host_info[conn.host]['gpu_memory_used'] = temp

    def gpt_memory_free_update(self):
        used_memory: GroupResult = self.prod.run('nvidia-smi --query-gpu=memory.free --format=csv', hide=True)
        for conn, result in used_memory.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                temp.append(float(res[i][:-4]))
            self.host_info[conn.host]['gpu_memory_free'] = temp

    def __init__(self, host_info):
        self.host_info = host_info
        self.pod_info = dict()
        self.connect()
        self.total_gpu_memory_set()

        self.update()

    def update(self):
        self.gpu_memory_used_update()
        self.gpu_utils_update()
        self.gpt_memory_free_update()

    def get_host_info(self):
        return self.host_info


if __name__ == '__main__':
    # podGPUMonitor = podGPUMonitor(host_info, 'ResNet')
    start1 = time.time()
    gpuMonitor = GPUMonitor(host_info)

    print("..")
