from fabric import Connection
from fabric.runners import Result
from fabric import SerialGroup as Group
from fabric.group import GroupResult
import re
import json
from datetime import datetime
import time
import kubernetes as k8s
from kubernetes import client
import multiprocessing
from multiprocessing import Manager
import gc

# from fabric import ThreadingGroup as Group

hostinfo = dict()
# hostinfo['192.168.1.77'] = {'user': 'root', 'password': '123456', 'gpu_num': 4}
# hostinfo['192.168.1.66'] = {'user': 'root', 'password': 'lc123456', 'gpu_num': 1}
# hostinfo = OrderedDict()
hostinfo['192.168.1.77'] = {'user': 'root', 'password': '123456', 'gpu_num': 4, 'node_id': 1, 'node_name': 'master', 'GPUmemory_util': [0, 0, 0, 0]}
hostinfo['192.168.1.66'] = {'user': 'root', 'password': 'lc123456', 'gpu_num': 1, 'node_id': 2, 'node_name': 'node1', 'GPUmemory_util': [0]}
connections = list()

k8s.config.load_kube_config(config_file="./kubeconfig.yaml.yaml")
k8sCoreV1api = client.CoreV1Api()  # 获取API的版本对象


class gpuMonitor():
    def connect(self):
        for host in self.hostinfo:
            conn = Connection(host, user=self.hostinfo[host]['user'], port=22,
                              connect_kwargs={"password": self.hostinfo[host]['password']})
            conn.run('hostname')  # 这行不能忽略
            connections.append(conn)

        self.prod = Group.from_connections(connections)
        return 0

    def total_gpumemory_set(self):
        total_memory: GroupResult = self.prod.run('nvidia-smi --query-gpu=memory.total --format=csv',hide=True)
        for conn, result in total_memory.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                temp.append(float(res[i][:-4]))
            self.hostinfo[conn.host]['gpumemory_total'] = temp
            # self.hostinfo[conn.host]['gpumemory_free'] = temp

    def gpu_utils_update(self):
        gpu_results: GroupResult = self.prod.run('nvidia-smi --query-gpu=utilization.gpu --format=csv',hide=True)
        # 分别进行字符串解析
        for conn, result in gpu_results.items():
            temp = []
            res = result.stdout.split('\n')
            for i in range(1, len(res) - 1):
                temp.append(float(res[i][:-2]))
            self.hostinfo[conn.host]['gpu_utils'] = temp

    def get_bandwidth(self):
        print('get')
        # 上行带宽：客户端->服务端
        # 下行带宽：服务端->客户端
        Server_Result: Result = self.prod[0].run('iperf -s -P 2', hide=True)  # 启动服务端
        Client_Result: Result = self.prod[1].run('iperf -c master -t 3 -d', hide=True)  # 启动客户端
        res = Client_Result.stdout.split('\n')
        bandwidth = [float(b.split('   ')[-1].split(' ')[0]) for b in res[-3:-1]]
        print(bandwidth)
        self.hostinfo[connections[0].host]['bandwidth'] = bandwidth[::-1]
        self.hostinfo[connections[1].host]['bandwidth'] = bandwidth

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
                    pod_info[pod_name]['epoch'], pod_info[pod_name]['time'] = self.get_pod_logs(pod_name, i-1)

                # pod_info_list.append(pod_info)
        self.pod_info = pod_info
        return pod_info

    def get_pod_logs(self, podname, node_id):
        log_name_pattern = r"(.*)-(.*)-(.*)"
        # 同一个job的不同worker运行时，日志（几乎）一样，且不同节点上的worker编号均从0开始
        matchObj = re.match(log_name_pattern, podname[:-1]+'0')
        jobname = matchObj.group(1)
        # if jobname == 'dcgan' or jobname == 'dcgan':
        #     return -1, None
        filename = '{}-{}'.format(matchObj.group(2), matchObj.group(3))
        if node_id == 0:
            filepath = '/27T/nfs/kubeflow/{}/{}.txt'.format(jobname, filename)
            logsResult: Result = self.prod[0].run('tail -n 1 {}'.format(filepath), hide=True)
        else:
            filepath = '/home/nfs/kubeflow/{}/{}.txt'.format(jobname, filename)
            logsResult: Result = self.prod[1].run('tail -n 1 {}'.format(filepath), hide=True)
        tup = logsResult.stdout.split(', ')
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
        t1 = time.time()
        self.gpumemory_used_update()
        # print('gpumemory_used_update:', time.time() - t1, 's')
        t2 = time.time()
        self.gpu_utils_update()
        # print('gpu_utils_update:', time.time() - t2, 's')
        t3 = time.time()
        self.get_pod_gpu_info()
        # print('get_pod_gpu_info:', time.time()-t3, 's')

    def get_host_info(self):
        return self.hostinfo


def start_monitor(share_info):
    gpu_monitor = gpuMonitor(hostinfo)
    share_info['host'] = gpu_monitor.get_host_info()
    share_info['job'] = gpu_monitor.get_pod_gpu_info()
    while True:
        gpu_monitor.update()
        share_info['host'] = gpu_monitor.get_host_info()
        share_info['job'] = gpu_monitor.get_pod_gpu_info()
        time.sleep(5)


def getUseNode(k8sCoreV1api):
    """
    # 获取所有节点
    :param k8sCoreV1api: k8sApi
    :return: useNodeName: 所有节点的节点名称
    """
    nodeInstance = k8sCoreV1api.list_node()
    useNodeName = []
    for i in nodeInstance.items:
        if i.status.conditions[-1].status == "True" and i.status.conditions[-1].type == "Ready":
            useNodeName.append(i.metadata.name)
    return useNodeName


def get_node_info(node, host_state, return_dict):
    temp_node_info = dict()
    node_ip_dict = {'master': '192.168.1.77', 'node1': '192.168.1.66'}
    temp_node_info['node_id'] = hostinfo[node_ip_dict[node]]['node_id']
    temp_node_info['GPU_num'] = hostinfo[node_ip_dict[node]]['gpu_num']
    temp_node_info['GPU_util'] = host_state[node_ip_dict[node]]['gpu_utils']
    temp_node_info['GPUmemory_util'] = host_state[node_ip_dict[node]]['gpumemory_used']
    temp_node_info['GPUmemory_total'] = host_state[node_ip_dict[node]]['gpumemory_total']
    return_dict[node] = temp_node_info


def main(gpu_monitor):
    useNodeName = getUseNode(k8sCoreV1api)  # 获取可用节点
    # job_name_list = ['vgg-1-3-30-16', 'vgg-1-2-50-32', 'vgg-1-4-40-16', 'resnet-2-2-30-8', 'resnet-2-3-50-16',
    #                  'resnet-2-4-40-8', 'unet-3-2-30-8', 'unet-3-3-50-16', 'unet-3-4-40-8', 'cnn-4-4-10-16',
    #                  'cnn-4-5-20-32', 'cnn-4-3-10-32']

    hostinfo['192.168.1.77']['GPUmemory_util'] = [0, 0, 0, 0]
    hostinfo['192.168.1.66']['GPUmemory_util'] = [0]
    # start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # run(job_name_list, log_path, start_time, useNodeName, log, execution_time, job_tracker, gpu_monitor, agent)
    for j in range(1000):
        # curr_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        host_state = gpu_monitor['host'].copy()
        man = Manager()
        return_dict = man.dict()
        jobs = []
        for node in useNodeName:
            p = multiprocessing.Process(target=get_node_info, args=(node, host_state, return_dict))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        node_info = return_dict.copy()
        del man, return_dict, jobs
        gc.collect()  # 垃圾回收
        print(j, node_info)
        time.sleep(1)



if __name__ == '__main__':
    # podGPUMonitor = GPUMonitor(hostinfo, 'ResNet')
    start1 = time.time()
    gpuMonitor = gpuMonitor(hostinfo)
    print("初始化时间: ", time.time()-start1, 's')
    start2 = time.time()
    gpuMonitor.update()
    print("更新时间：", time.time() - start2, 's')
    start3 = time.time()
    filepath = '/27T/nfs/kubeflow/{}/{}.txt'.format('cnn-4-5-20-32', 'worker-0')
    logsResult: Result = gpuMonitor.prod[0].run('tail -n 1 {}'.format(filepath), hide=True)
    print("读文件时间：", time.time() - start3, 's')
    # print(gpuMonitor.get_bandwidth(), "测带宽时间：", time.time()-start2, 's')

    manager = Manager()
    share_info = manager.dict()

    p1 = multiprocessing.Process(target=start_monitor, args=(share_info,))
    p1.daemon = True  # 设置为守护进程
    p2 = multiprocessing.Process(target=main, args=(share_info,))
    p1.start()
    time.sleep(40)
    p2.start()
    p2.join()
    gc.collect()



