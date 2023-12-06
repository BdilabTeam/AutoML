import paramiko
from fabric import Connection
from fabric.runners import Result
from fabric import SerialGroup as Group
from fabric.group import GroupResult
import re
import json
from datetime import datetime
import time

# from fabric import ThreadingGroup as Group

hostinfo = dict()
hostinfo['10.170.23.190'] = {'user': 'root', 'password': '123456', 'gpu_num': 2, 'node_id': 1, 'node_name': 'master', 'GPUmemory_util': [0, 0]}

connections = list()

class gpuMonitor():
    def connect(self):
        for host in self.hostinfo:

            conn = Connection(host, user=self.hostinfo[host]['user'], port=22,
                              connect_kwargs={"password": self.hostinfo[host]['password']},
                              connect_timeout=200)
            conn.run('hostname')  # 这行不能忽略
            connections.append(conn)

        self.prod = Group.from_connections(connections)
        return 0

    def total_gpumemory_set(self):
        total_memory: GroupResult = self.prod.run('nvidia-smi --query-gpu=memory.total --format=csv', hide=True)
        for conn, result in total_memory.items():
            temp = []
            # 用换行符进行分割的原因是每行代表了一个GPU的信息
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                # 为什么是 :-4? 举例:"15360 MiB"
                temp.append(float(res[i][:-4]))
            self.hostinfo[conn.host]['gpumemory_total'] = temp

    def gpu_utils_update(self):
        gpu_results: GroupResult = self.prod.run('nvidia-smi --query-gpu=utilization.gpu --format=csv', hide=True)
        # 分别进行字符串解析
        for conn, result in gpu_results.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                # 35 %
                temp.append(float(res[i][:-2]))
            self.hostinfo[conn.host]['gpu_utils'] = temp

    def get_bandwidth(self):
        print('get')
        # 上行带宽：客户端->服务端
        # 下行带宽：服务端->客户端
        Server_Result: Result = self.prod[0].run('iperf -s -P 2', hide=True)  # 启动服务端
        Client_Result: Result = self.prod[0].run('iperf -c master -t 3 -d', hide=True)  # 启动客户端
        res = Client_Result.stdout.split('\n')
        print(Client_Result.stdout)
        bandwidth = [float(b.split('   ')[-1].split(' ')[0]) for b in res[-3:-1]]
        print(bandwidth)
        self.hostinfo[connections[0].host]['bandwidth'] = bandwidth[::-1]
        self.hostinfo[connections[0].host]['bandwidth'] = bandwidth

    def gpumemory_used_update(self):
        used_memory: GroupResult = self.prod.run('nvidia-smi --query-gpu=memory.used --format=csv', hide=True)
        for conn, result in used_memory.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                temp.append(float(res[i][:-4]))
            self.hostinfo[conn.host]['gpumemory_used'] = temp

    def get_pod_gpu_info(self):
        name_pattern = r"k8s_tensorflow_(.*)_dljobs_(.*)"
        pod_infos: GroupResult = self.prod.run('/root/bin/test 2> /dev/null', hide=True)
        # pod_info_list = []
        pod_info = dict()
        for conn, result in pod_infos.items():

            res = result.stdout.split('\n')

            for i in range(1, len(res) - 1):
                items = res[i].split('\t\t')

                # 有时会报items【2】超出索引的错误
                if len(items) > 2:
                    pod_name = re.match(name_pattern, items[2]).group(1)
                    pod_info[pod_name] = dict()
                    pod_info[pod_name]['gpumemory_used'] = float(items[3].split('MiB')[0])
                    pod_info[pod_name]['gpu_id'] = int(items[4])
                    pod_info[pod_name]['epoch'], pod_info[pod_name]['time'] = self.get_pod_logs(pod_name, i)

                # pod_info_list.append(pod_info)
        self.pod_info = pod_info
        return pod_info

    def get_pod_logs(self, podname, node_id):
        log_name_pattern = r"(.*)-(.*)-(.*)"
        # 同一个job的不同worker运行时，日志（几乎）一样，且不同节点上的worker编号均从0开始
        matchObj = re.match(log_name_pattern, podname[:-1] + '0')
        jobname = matchObj.group(1)
        # if jobname == 'dcgan' or jobname == 'dcgan':
        #     return -1, None
        filename = '{}-{}'.format(matchObj.group(2), matchObj.group(3))
        if node_id == 1:
            filepath = '/data/nfs/{}/{}.txt'.format(jobname, filename)
            logsResult: Result = self.prod[0].run('tail -n 1 {}'.format(filepath), hide=True)
        else:
            filepath = '/home/nfs/kubeflow/{}/{}.txt'.format(jobname, filename)
            logsResult: Result = self.prod[1].run('tail -n 1 {}'.format(filepath), hide=True)
        tup = logsResult.stdout.split(', ')
        # print(tup)
        if len(tup) < 2:
            return -1, None
        else:
            return int(tup[0]), tup[1].split('.')[0]

    def __init__(self, hostinfo):
        self.hostinfo = hostinfo
        self.pod_info = dict()
        self.connect()
        self.total_gpumemory_set()
        # self.gpumemory_used_update()
        # self.gpu_utils_update()
        # self.get_pod_gpu_info()
        self.update()

    def update(self):
        self.gpumemory_used_update()
        self.gpu_utils_update()
        self.get_pod_gpu_info()

    def get_host_info(self):
        return self.hostinfo


if __name__ == '__main__':
    # podGPUMonitor = podGPUMonitor(hostinfo, 'ResNet')
    start1 = time.time()
    gpuMonitor = gpuMonitor(hostinfo)
    print("初始化时间: ", time.time() - start1, 's')
    start2 = time.time()
    print(gpuMonitor.get_bandwidth(), "测带宽时间：", time.time() - start2, 's')
