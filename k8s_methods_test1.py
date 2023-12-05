#!/user/bin/env python
# -*- coding: UTF-8 -*-

"""
@author: liu wenjing
@create: 2021/9/24 15:34
"""
import torch.multiprocessing
import kubernetes as k8s
from kubernetes import client  # kubernets官方维护的python客户端库
import json
import numpy as np
import schedulingAlgo  # py文件
from monitoring import gpuMonitor  # py文件
import os
import datetime
import time
from collections import OrderedDict  # 有序字典，记住字典的插入顺序
import RLschedulingAlgo
from job_submission import jobSubmission
import numpy.random as random  # 迷惑我
import multiprocessing
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
import gc
from performance_online import Performance, add_time_interval
from monitor_writer import monitor_writer
import job_name_constants
from tensorboardX import SummaryWriter
from subprocess import call
import pickle
from yaml_generate import generate
import my_logging

if os.path.exists("main.log"):
    os.remove("main.log")
mylog = my_logging.MyLogger(filename="main.log", level="debug")

nToCoreNum = 1000 * 1000 * 1000  # cpu单位换算
nToMi = 1024

# 建立node_name和ip地址之间的映射
node_ip_dict = {'master': '10.170.23.190'}
# 关于集群节点的信息
hostinfo = OrderedDict()
hostinfo['10.170.23.190'] = {'user': 'root', 'password': '123456', 'gpu_num': 2, 'node_id': 1, 'node_name': 'master',
                             'GPUmemory_util': [0, 0]}
dim_dict = {'max_node_num': 1, 'max_GPU_num': 2, 'max_worker_num': 8, 'max_job_type': 8,
            'max_job_running': 20, 'max_job_gpu': 16}
# node_dict = {1: 'master', 2: 'node1', 3: 'node2', 4: 'node3'}
node_dict = {1: 'master'}
reward_interval = 1
job_resource_demand = {'vgg-16': {'GPUmem_demand': 5000, 'GPUutil_demand': 90, 'time_per_epoch': 80, 'time_weight': 24},
                       # MiB 90%
                       'vgg-32': {'GPUmem_demand': 5000, 'GPUutil_demand': 90, 'time_per_epoch': 65, 'time_weight': 40},
                       'resnet-8': {'GPUmem_demand': 5000, 'GPUutil_demand': 35, 'time_per_epoch': 85,
                                    'time_weight': 40},
                       'resnet-16': {'GPUmem_demand': 5000, 'GPUutil_demand': 30, 'time_per_epoch': 45,
                                     'time_weight': 120},
                       'unet-8': {'GPUmem_demand': 4500, 'GPUutil_demand': 45, 'time_per_epoch': 50, 'time_weight': 24},
                       'unet-16': {'GPUmem_demand': 4500, 'GPUutil_demand': 60, 'time_per_epoch': 30,
                                   'time_weight': 60},
                       'cnn-16': {'GPUmem_demand': 1100, 'GPUutil_demand': 30, 'time_per_epoch': 140,
                                  'time_weight': 33},
                       'cnn-32': {'GPUmem_demand': 1100, 'GPUutil_demand': 35, 'time_per_epoch': 66, 'time_weight': 80},
                       'rnn-8': {'GPUmem_demand': 1300, 'GPUutil_demand': 35, 'time_per_epoch': 20, 'time_weight': 80},
                       'rnn-16': {'GPUmem_demand': 1300, 'GPUutil_demand': 45, 'time_per_epoch': 25,
                                  'time_weight': 240},
                       'cnnl-16': {'GPUmem_demand': 1100, 'GPUutil_demand': 50, 'time_per_epoch': 30,
                                   'time_weight': 80},
                       'cnnl-32': {'GPUmem_demand': 1100, 'GPUutil_demand': 60, 'time_per_epoch': 22,
                                   'time_weight': 240},
                       'dcgan-16': {'GPUmem_demand': 1400, 'GPUutil_demand': 60, 'time_per_epoch': 22,
                                    'time_weight': 40},
                       'dcgan-32': {'GPUmem_demand': 1400, 'GPUutil_demand': 60, 'time_per_epoch': 22,
                                    'time_weight': 40},
                       'seq2seq-16': {'GPUmem_demand': 1100, 'GPUutil_demand': 60, 'time_per_epoch': 30,
                                      'time_weight': 40},
                       'seq2seq-32': {'GPUmem_demand': 1100, 'GPUutil_demand': 60, 'time_per_epoch': 30,
                                      'time_weight': 40},
                       }
# 加载集群的配置信息——配置文件的路径， 这个函数在客户端连接到集群之前被调用
# 配置信息被封装在配置文件中，而不是在代码中
k8s.config.load_kube_config(config_file="./kubeconfig.yaml")  # 加载配置文件中的信息，将其设置为客户端库的默认配置

# 访问k8s集群的核心API资源的客户端对象
k8sCoreV1api = client.CoreV1Api()  # 获取与集群交互的对象
api_client = client.ApiClient()  # 创建客户端实例
namespace = "dljobs"  # 指定pod所在的命名空间
output_dir = './log/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#
# job_time = {'0': {'weight': 3, 'model_type': 'resnet'},
#            '1': {'weight': 1, 'model_type': 'rnn'},
#            '2': {'weight': 3, 'model_type': 'vgg'},
#            '3': {'weight': 1.5, 'model_type': 'unet'},
#            '4': {'weight': 4, 'model_type': 'cnn'},
#            '5': {'weight': 1, 'model_type': 'cnnl'},
#            '6': {'weight': 1, 'model_type': 'dcgan'},
#            '7': {'weight': 1, 'model_type': 'seq2seq'}}

# 获取所有可用的节点，返回节点名称列表
def getUseNode(k8sCoreV1api):
    nodeInstance = k8sCoreV1api.list_node()  # 列出集群中所有节点的信息，包括节点的名称，ip，资源利用率等
    useNodeName = []
    for i in nodeInstance.items:  # for i in list[V1Node]
        # 可用的节点：the node is healthy and ready to accept pods
        if i.status.conditions[-1].status == "True" and i.status.conditions[-1].type == "Ready":
            useNodeName.append(i.metadata.name)  # V1Node.metadata.name
    return useNodeName


# 获取指定命名空间内所有pod的信息
def getSchedulingPod(k8sCoreV1api, namespace):
    podInstance = k8sCoreV1api.list_namespaced_pod(namespace)
    # 该函数在kubernetes.client.api.core_V1_api.CoreV1Api中
    scheduledPodName = []
    for i in podInstance.items:  # for i in list[V1Pod]
        # pod期望状态:spec，pod当前实际状态：status
        if i.status.phase == 'Pending' and i.spec.node_name is None:
            scheduledPodName.append(i.metadata.name)
    return scheduledPodName


# binding：将指定的(Pod)绑定到特定的(Node)上
def podBinding(k8sCoreV1api, podName, nodeName, namespace, gpu_index):
    target = client.V1ObjectReference()  # kubernetes.client.models.v1_object_reference中定义的类
    target.kind = "Node"
    target.api_version = "v1"
    target.name = nodeName

    # 所有持久化资源必须具备元数据 元数据通常包括与资源对象（pod、node）相关的元信息
    meta = client.V1ObjectMeta()
    meta.name = podName
    # annotations属性是dict(str,str)
    # 可以使用annotations将任意非标识性元数据附加到对象上，注释不用于识别和选择对象。
    # TODO master里'ALIYUN_COM_GPU_MEM_DEV': "40" 我之前改的 之前是一个很小的值 现在要执行任务 就和else同步了
    if nodeName == 'master':
        meta.annotations = {'ALIYUN_COM_GPU_MEM_ASSIGNED': "false",
                            'ALIYUN_COM_GPU_MEM_ASSUME_TIME': str(time.time_ns()),
                            'ALIYUN_COM_GPU_MEM_DEV': "24", 'ALIYUN_COM_GPU_MEM_IDX': str(gpu_index),
                            # master要执行任务，所以增加MEM_DEV
                            'ALIYUN_COM_GPU_MEM_POD': "1"}

    else:
        meta.annotations = {'ALIYUN_COM_GPU_MEM_ASSIGNED': "false",
                            'ALIYUN_COM_GPU_MEM_ASSUME_TIME': str(time.time_ns()),
                            'ALIYUN_COM_GPU_MEM_DEV': "40", 'ALIYUN_COM_GPU_MEM_IDX': str(gpu_index),
                            'ALIYUN_COM_GPU_MEM_POD': "1"}
    # ALIYUN_COM_GPU_MEM_ASSIGNED： 一定是false才可以调度
    # ALIYUN_COM_GPU_MEM_ASSUME_TIME: 时间戳，随便设置  time.time_ns()获取当前时间的纳秒级别精确度的时间戳，返回一个整数
    # ALIYUN_COM_GPU_MEM_DEV: 被分配的显卡的显存
    # ALIYUN_COM_GPU_MEM_IDX: 被分配的显卡的id
    # ALIYUN_COM_GPU_MEM_POD： 请求的显存

    # 创建一个k8s的绑定对象body：调度程序将pod与指定节点绑定在一起
    body = client.V1Binding(target=target)
    body.target = target  # V1ObjectReference类型
    body.metadata = meta  # V1ObjectMeta类型
    # 下面一定会抛出异常（不影响正常运行），所以必须加try，不然程序就终止
    try:
        # 在指定的命名空间中创建一个绑定对象
        # TODO 会报错409  pod已经被调度到节点 冲突
        # TODO 现在不会报错
        k8sCoreV1api.create_namespaced_binding(namespace, body, _preload_content=False)
        # k8sCoreV1api.create_namespaced_binding(namespace, body)  # 将pod调度到node
        return True
    except Exception as e:
        print('exception' + str(e))
        return False


# 根据调度策略进行pod调度
def podScheduling(k8sCoreV1api, policy, job_tracker, job_epoch_records):
    # 遍历policy中的每个Pod
    for podName in policy:
        print("开始调度")
        nodeName = policy[podName]['node_name']
        namespace = policy[podName]['namespace']

        # 在这里对job_tracker字典中的两个列表进行添加，记录gpu_id和node_id的信息！！！！！！
        # job_tracker(以job_id作为键)
        if nodeName:
            job_tracker[policy[podName]['job_id']]['gpu_id'].append(policy[podName]['gpu_index'])
            job_tracker[policy[podName]['job_id']]['node_id'].append(policy[podName]['node_id'])
            # 创建job_epoch_records字典，键是job_id，值是OrderedDict()
            job_epoch_records[policy[podName]['job_id']] = OrderedDict()

            key = policy[podName]['job_id'].split('-')  # 字符串->字符串的列表
            # job_resource_reference是由key中第一个元素和第五个元素组成的字符串，作为job_resource_demand字典的索引
            job_resource_reference = key[0] + '-' + key[4]
            # 更新节点上的GPU内存利用率：将作业调度到节点的某一gpu上之后，对应的GPUmemory_util要增加
            # job_resource_demand 获取job所需的gpu内存，更新hostinfo
            hostinfo[node_ip_dict[nodeName]]['GPUmemory_util'][policy[podName]['gpu_index']] += job_resource_demand[job_resource_reference]['GPUmem_demand']
            re = podBinding(k8sCoreV1api, podName, nodeName, namespace, policy[podName]['gpu_index'])
            print("完成一个pod的调度")
    return policy


# 将以字符串表示的内存大小转换为以Mi为单位的浮点数 Ki是k8s默认的单位
def memory_to_mi(memory_string):  # memory_to_mi的参数已经去掉i
    order = memory_string[-1]
    # nToMi = 1024
    if order == 'K':
        return float(memory_string[:-1]) / nToMi  # 字符串切片 去掉最后一个表示单位的字符
    if order == 'M':
        return float(memory_string[:-1])
    if order == 'G':
        return float(memory_string[:-1]) * nToMi


# 获取节点的cpu核数、内存容量、所占用的cpu核数、所占用的内存容量
def getNodeResource(k8sCoreV1api, api_client, nodeName):
    # read_node_status(nodeName) read status of the specified Node 读取指定节点的状态信息，返回V1Node类型
    # V1Node.status是V1NodeStatus，其中的capacity字段是dict(str,str)，表示节点的容量信息
    # 内存容量单位：以2的幂次方计算，MiB和MB有区别，MiB表示以2为底的二进制容量
    # CPU资源以cpu为单位，1cpu=1000m,1m=1000*1000n
    CPU_core = int(k8sCoreV1api.read_node_status(nodeName).status.capacity['cpu'])  # 返回cpu核数
    memory_size = memory_to_mi(k8sCoreV1api.read_node_status(nodeName).status.capacity['memory'][:-1])

    # url是一个包含 占位符{} 的字符串，用于动态插入节点名称，这个字符串表示k8s中节点度量数据的路径
    url = '/apis/metrics.k8s.io/v1beta1/nodes/{}'.format(nodeName)
    '''
    call_api用于执行API请求，接受多个参数，指定请求的细节
    url：要访问的API资源的路径
    GET:表示HTTP请求方法，要执行的请求是获取资源、数据
    auth_settings=['BearerToken']：身份验证设置，指定使用令牌
    response_type='json'：期望的响应类型，希望获得json格式的响应数据
    _preload_content=False：告诉客户端不要立即加载响应内容，而是等待后续的请求来获取响应的内容
    '''
    # metric-server，是一个集群级别的资源指标收集器，提供cpu、内存监控接口查询，对外通过Metric API暴露给外部访问
    metrics = api_client.call_api(url, 'GET', auth_settings=['BearerToken'], response_type='json',
                                  _preload_content=False)
    # 将字节数据解码为UTF-8编码的字符串
    response = metrics[0].data.decode('utf-8')
    # json.loads将json格式的字符串转换为python字典
    response = json.loads(response)
    # cpu在k8s中的单位是m
    CPU_usage = float(response['usage']['cpu'].split('n')[0]) / nToCoreNum  # nToCoreNum = 1000*1000*1000
    memory_usage = memory_to_mi(response['usage']['memory'].split('i')[0])
    return CPU_core, CPU_usage, memory_size, memory_usage


# 获取pod所占用的cpu核数和内存容量
def getPodResource(api_client, namespace, podName):
    url = '/apis/metrics.k8s.io/v1beta1/namespaces/{0}/pods/{1}'.format(namespace, podName)
    metrics = api_client.call_api(url, 'GET', auth_settings=['BearerToken'], response_type='json',
                                  _preload_content=False)
    response = metrics[0].data.decode('utf-8')
    response = json.loads(response)
    CPU_usage = float(response['containers'][0]['usage']['cpu'].split('n')[0]) / nToCoreNum
    memory_usage = memory_to_mi(response['containers'][0]['usage']['memory'].split('i')[0])
    return CPU_usage, memory_usage


# 获取pod的创建时间戳
def getPodInfo(k8sCoreV1api, namespace, podName):
    # read status of the specified Pod
    res = k8sCoreV1api.read_namespaced_pod_status(podName, namespace)  # res是V1Pod
    # Pod创建时间为UTC，需要加上八小时，和北京时间时间相同
    # V1ObjectMeta的属性creation_timestamp（创建时间戳）是datatime类型
    creation_timestamp = res.metadata.creation_timestamp + datetime.timedelta(hours=8)
    # 将日期和时间格式化为字符串
    return creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')


# 节点GPU相关信息
def get_node_info(node, host_state, return_dict):  # node是node_name

    # 创建一个临时的空字典，存储节点的信息
    temp_node_info = dict()
    # 根据nodeName获取node_id
    temp_node_info['node_id'] = hostinfo[node_ip_dict[node]]['node_id']
    temp_node_info['GPU_num'] = hostinfo[node_ip_dict[node]]['gpu_num']

    temp_node_info['GPU_util'] = host_state[node_ip_dict[node]]['gpu_utils']  # gpu使用率
    temp_node_info['GPUmemory_util'] = host_state[node_ip_dict[node]]['gpumemory_used']  # 已使用的显存
    temp_node_info['GPUmemory_total'] = host_state[node_ip_dict[node]]['gpumemory_total']  # 总共的显存
    # 将获取的信息存储在return_dict中，该字典以nodeName为键
    return_dict[node] = temp_node_info


# 用于作业提交时的信息字典，它的键是node_index，即节点的索引，以0开始
submission_info = dict()
submission_info[0] = {'ip': '10.170.23.190', 'user': 'root', 'password': '123456', 'dir': '/home/bdilab/dl2/jobs'}


# 作业调度及性能分析
def run(job_name_list, log_path, start_time, useNodeName, log, execution_time, job_tracker, share_info, agent,
        performance, main_writer, submission_dict):
    # TODO 为什么再次打乱作业顺序
    random.shuffle(job_name_list)
    # submission_info 是 host_info
    submission = jobSubmission(submission_info, job_name_list)  # 创建一个作业提交对象

    # wait monitor
    time.sleep(20)

    submission.submit_list(node_index=0, name_list=submission_dict[0])  # 先提交0分组的作业
    # 提交作业的动作：create -f .yaml
    job_epoch_records = OrderedDict()
    finish_cnt = 0
    running_jobs = OrderedDict()

    for j in range(10000000):  # j代表时间步的变化
        # 这段代码检查是否需要在时间步j提交新的作业
        if j > 0 and submission_dict.__contains__(j):
            submission.submit_list(0, submission_dict[j])

        # 以时间步为键，然后在特定时间步下，以node_id为键，又有三个子字典
        running_jobs[main_writer.get_step()] = OrderedDict({1: OrderedDict(), 2: OrderedDict(), 3: OrderedDict()})
        curr_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # share_info['host'] = hostinfo  # 主机的状态信息

        # 通过monitor进程不断地更新和监听
        host_state = share_info['host'].copy()
        man = Manager()  #
        return_dict = man.dict()  # 创建共享字典

        # 以下代码的作用是为每个节点创建一个进程 并行地获取不同节点的信息，将结果保存在return_dict字典中。
        # jobs = []
        # for node in useNodeName:
        #     # 创建进程并添加到jobs列表中
        #     # 指定该进程要运行的目标函数是get_node_info，传递给目标函数的参数是args
        #
        #     p = multiprocessing.Process(target=get_node_info, args=(node, host_state, return_dict))
        #     jobs.append(p)
        #     p.start() # 启动进程，使其执行目标函数

        # 获取node的信息(node_id,GPU_num,GPU_util,GPUmemory_util,GPUmemory_total) 返回给return_dict
        get_node_info(useNodeName[0], host_state, return_dict)
        # wait for monitor
        time.sleep(10)

        # for proc in jobs:
        #     proc.join() # 等待所有的子进程执行完成，然后再执行主进程
        # 将不同节点的信息整理成node_info字典
        node_info = return_dict.copy()
        del man, return_dict
        gc.collect()  # 垃圾回收

        for node in node_info:
            # write_usage(self, node, mem_usage, gpu_usage) 其中mem_usage和gpu_usage都是比率

            #   temp_node_info['GPUmemory_util'] = host_state[node_ip_dict[node]]['gpumemory_used']  # 已使用的显存
            main_writer.write_usage(node, np.mean(np.array(node_info[node]['GPUmemory_util'])) / np.mean(
                np.array(node_info[node]['GPUmemory_total'])),
                                    np.mean(np.array(node_info[node]['GPU_util'])))

            # 预留显存  [5000, 0]

            node_info[node]['GPUmemory_util'] = hostinfo[node_ip_dict[node]]['GPUmemory_util']

        job_info = OrderedDict()
        # share_info['job']由get_pod_gpu_info()函数获取，pod_info_dict以pod_name为键

        pod_info_dict = share_info['job'].copy()

        # 遍历的是dljobs命名空间里所有的pod
        for i in k8sCoreV1api.list_namespaced_pod(namespace).items:  # i是V1Pod
            # temp_pod_info中的信息是从k8s中pod yaml文件中获取的
            temp_pod_info = OrderedDict()
            temp_pod_info['namespace'] = i.metadata.namespace
            temp_pod_info['phase'] = i.status.phase
            job_id = i.metadata.owner_references[0].name  # 形如cnnl-6-2-10-16

            # pod 的资源请求,待填充
            # temp_pod_info['CPU_demand'] = 0
            # temp_pod_info['memory_demand'] = 0
            # temp_pod_info['job_id'] = job_id

            # 正式实验中设置格式job_id为 模型名称-模型id-worker数量-epoch数量-batch_size  例'dcgan-7-2-10-16'
            job_meta = job_id.split('-')
            temp_pod_info['model_name'] = job_meta[0]
            temp_pod_info['job_type'] = int(job_meta[1]) - 1  # 区分作业的类型 为什么-1 大概因为索引从0开始
            temp_pod_info['worker_num'] = int(job_meta[2])
            temp_pod_info['epoch_num'] = int(job_meta[3])
            temp_pod_info['batch_size'] = int(job_meta[4])
            # creation_timestamp是datetime.datetime(2023, 11, 28, 8, 20, 54, tzinfo=tzutc())
            temp_pod_info['creation_time'] = (i.metadata.creation_timestamp + datetime.timedelta(hours=8)).strftime(
                '%Y-%m-%d %H:%M:%S')
            resource_demand_reference = job_meta[0] + '-' + job_meta[4]  # name-batch_size
            temp_pod_info['GPUutil_demand'] = job_resource_demand[resource_demand_reference]['GPUutil_demand']
            temp_pod_info['GPUmem_demand'] = job_resource_demand[resource_demand_reference]['GPUmem_demand']
            temp_pod_info['time_per_epoch'] = job_resource_demand[resource_demand_reference]['time_per_epoch']
            # 判断job_id是否为job_info的一个键，如果是则说明字典中已经创建了这个作业的子字典

            # job_info以job_id为键
            # 同一个job由多个worker
            if job_info.__contains__(job_id):
                job_info[job_id]['pod_list'].append(i.metadata.name)
                job_info[job_id]['phase'].append(temp_pod_info['phase'])
            # 如果不包含就创建一个新的字典存储相关信息
            else:
                job_info[job_id] = OrderedDict()
                job_info[job_id]['job_index'] = len(job_info)
                job_info[job_id]['namespace'] = i.metadata.namespace
                job_info[job_id]['pod_list'] = []
                job_info[job_id]['pod_list'].append(i.metadata.name)
                job_info[job_id]['phase'] = []
                job_info[job_id]['phase'].append(temp_pod_info['phase'])
                job_info[job_id]['worker_num'] = temp_pod_info['worker_num']
                job_info[job_id]['job_type'] = temp_pod_info['job_type']
                job_info[job_id]['GPUutil_demand'] = temp_pod_info['GPUutil_demand']
                job_info[job_id]['GPUmem_demand'] = temp_pod_info['GPUmem_demand']
                job_info[job_id]['time_per_epoch'] = temp_pod_info['time_per_epoch']
                job_info[job_id]['batch_size'] = temp_pod_info['batch_size']
                job_info[job_id]['epoch_num'] = temp_pod_info['epoch_num']
                job_info[job_id]['start_time'] = []
                job_info[job_id]['end_time'] = []
                job_info[job_id]['running_node'] = []
                job_info[job_id]['creation_time'] = []
                job_info[job_id]['corr_gpu_util'] = []
                job_info[job_id]['epoch'] = []
                job_info[job_id]['epoch_time'] = []
                job_info[job_id]['gpumemory_used'] = []
                job_info[job_id]['gpu_id'] = []

            if i.status.phase == "Succeeded":  # Pod中的所有容器都已经成功终止，不会重新启动
                job_info[job_id]['start_time'].append(
                    (i.status.start_time + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))

                # container_statuses是一个列表，包含与Pod相关联的所有容器的状态信息
                # terminated表示容器已经终止 finished_at是容器终止的时间戳
                if i.status.container_statuses[0].state.terminated.finished_at is not None:
                    job_info[job_id]['end_time'].append(
                        (i.status.container_statuses[0].state.terminated.finished_at + datetime.timedelta(
                            hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    job_info[job_id]['end_time'].append(None)

                job_info[job_id]['running_node'].append(hostinfo[i.status.host_ip]['node_id'])
                job_info[job_id]['corr_gpu_util'].append(None)
                job_info[job_id]['creation_time'].append(temp_pod_info['creation_time'])
                job_info[job_id]['epoch'].append(None)
                job_info[job_id]['epoch_time'].append(None)
                job_info[job_id]['gpumemory_used'].append(None)
                job_info[job_id]['gpu_id'].append(None)
                # execution_time是run函数的一个参数 记录运行结束的job的运行时间
                if job_info[job_id]['end_time'][0] and job_id not in execution_time:
                    execution_time[job_id] = (
                                datetime.datetime.strptime(job_info[job_id]['end_time'][0], '%Y-%m-%d %H:%M:%S')
                                - datetime.datetime.strptime(job_info[job_id]['creation_time'][0],
                                                             '%Y-%m-%d %H:%M:%S')).seconds
                    # 这段代码用于更新gpu内存的利用率
                    # gpu_id和node_id是一一对应的
                    for k in range(len(job_tracker[job_id]['gpu_id'])):
                        # 所以hostinfo中的值的修改时刻 会 在调度pod开始会+=，成功或失败后释放gpu资源，要-=
                        hostinfo[node_ip_dict[node_dict[job_tracker[job_id]['node_id'][k]]]]['GPUmemory_util'][
                            job_tracker[job_id]['gpu_id'][k]] -= job_info[job_id]['GPUmem_demand']
            else:
                # 容器没有Succeeded
                if i.status.phase == "Running":
                    job_info[job_id]['start_time'].append(
                        (i.status.start_time + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
                    job_info[job_id]['end_time'].append(None)  # 尚未终止，将结束时间设置为None
                    job_info[job_id]['running_node'].append(hostinfo[i.status.host_ip]['node_id'])
                    job_info[job_id]['creation_time'].append(temp_pod_info['creation_time'])
                    if pod_info_dict.__contains__(i.metadata.name):

                        # node_info[node_name]['GPU_util']包含了节点上每个gpu的利用率（可能是个字典或列表）
                        # GPU_util是一个百分数
                        job_info[job_id]['corr_gpu_util'].append(
                            node_info[hostinfo[i.status.host_ip]['node_name']]['GPU_util']
                            [pod_info_dict[i.metadata.name]['gpu_id']])
                        job_info[job_id]['epoch'].append(pod_info_dict[i.metadata.name]['epoch'])
                        job_info[job_id]['epoch_time'].append(pod_info_dict[i.metadata.name]['time'])  # epoch完成时间
                        job_info[job_id]['gpumemory_used'].append(pod_info_dict[i.metadata.name]['gpumemory_used'])
                        job_info[job_id]['gpu_id'].append(pod_info_dict[i.metadata.name]['gpu_id'])
                    else:
                        job_info[job_id]['corr_gpu_util'].append(None)
                        job_info[job_id]['epoch'].append(0)
                        job_info[job_id]['epoch_time'].append(None)
                        job_info[job_id]['gpumemory_used'].append(None)
                        job_info[job_id]['gpu_id'].append(None)

                if i.status.phase == "Failed":
                    job_info[job_id]['start_time'].append(
                        (i.status.start_time + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
                    if i.status.container_statuses and i.status.container_statuses[0].state.terminated.finished_at:
                        job_info[job_id]['end_time'].append(
                            (i.status.container_statuses[0].state.terminated.finished_at + datetime.timedelta(
                                hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
                    else:
                        job_info[job_id]['end_time'].append(None)

                    job_info[job_id]['running_node'].append(hostinfo[i.status.host_ip]['node_id'])
                    job_info[job_id]['corr_gpu_util'].append(None)
                    job_info[job_id]['creation_time'].append(temp_pod_info['creation_time'])
                    job_info[job_id]['epoch'].append(None)
                    job_info[job_id]['epoch_time'].append(None)
                    job_info[job_id]['gpumemory_used'].append(None)
                    job_info[job_id]['gpu_id'].append(None)

                    if job_info[job_id]['end_time'][0] and job_id not in execution_time:
                        execution_time[job_id] = (
                                    datetime.datetime.strptime(job_info[job_id]['end_time'][0], '%Y-%m-%d %H:%M:%S') -
                                    datetime.datetime.strptime(job_info[job_id]['start_time'][0],
                                                               '%Y-%m-%d %H:%M:%S')).seconds
                        for k in range(len(job_tracker[job_id]['gpu_id'])):
                            hostinfo[node_ip_dict[node_dict[job_tracker[job_id]['node_id'][k]]]]['GPUmemory_util'][
                                job_tracker[job_id]['gpu_id'][k]] -= \
                                job_info[job_id]['GPUmem_demand']

                if i.status.phase == "Pending":
                    job_info[job_id]['start_time'].append(None)
                    job_info[job_id]['end_time'].append(None)
                    job_info[job_id]['running_node'].append(None)
                    job_info[job_id]['corr_gpu_util'].append(None)
                    job_info[job_id]['creation_time'].append(temp_pod_info['creation_time'])
                    job_info[job_id]['epoch'].append(None)
                    job_info[job_id]['epoch_time'].append(None)
                    job_info[job_id]['gpumemory_used'].append(None)
                    job_info[job_id]['gpu_id'].append(None)
        # 遍历pod列表结束

        if j > 0:
            # job_epoch_records是一个以job_id为键的字典，值是另一个以时间步j为键的字典，表示在不同时间步的作业状态信息
            for job_id in job_epoch_records:
                if job_epoch_records[job_id] is not None and job_id in job_info and job_info[job_id]['epoch'][
                    0] is not None and \
                        any(phase == 'Running' for phase in job_info[job_id]['phase']):
                    # 如果以上条件都满足，说明该作业正在运行且具有相关的状态信息
                    # job_epoch_records
                    job_epoch_records[job_id][str(j)] = performance.state_extraction(node_info, job_info, job_id,
                                                                                     job_resource_demand)
        performance.update_model(num_epochs=10)  # 训练的总轮次: 10 更新模型 保存模型参数和训练样本到对应路径

        # 以下代码统计每个节点上正在运行的作业数量
        for job_id in job_info:  # 遍历job_info中的所有作业
            temp_phase = job_info[job_id]['phase']

            if all([phase == 'Running' for phase in temp_phase]):
                for running_node in job_info[job_id]['running_node']:
                    # running_jobs以时间步为键，表示在特定的时间步下跟踪作业数量
                    # 在每个时间步内，有一个字典以node_id为键，表示在特定节点(running_node)上正在运行的作业数量
                    # 这种结构允许在不同的时间步下、不同的节点上记录每个作业的运行数量
                    if job_id not in running_jobs[main_writer.get_step()][running_node]:  # rnning_node是node_id
                        running_jobs[main_writer.get_step()][running_node][job_id] = 0
                    running_jobs[main_writer.get_step()][running_node][job_id] += 1

            if (any([phase == 'Succeeded' for phase in temp_phase]) or any([phase == 'Failed' for phase in temp_phase])) \
                    and job_id in job_name_list and job_id in execution_time:
                # 如果满足以上条件，认为作业已完成
                finish_cnt += 1  # 记录已完成的作业数量

        epoch = 0
        running_num = 0
        reward = 0
        tuned_loss_list = []
        loss_list = []
        predict_list = []
        real_list = []

        for job_id in job_tracker:
            # temp_job是dict(2)
            temp_job = job_tracker[job_id].copy()
            # 检查是否包含键'gpu_id'，如果包含就删除 为什么删除？
            if 'gpu_id' in temp_job:
                del temp_job['gpu_id']  # 删除字典中的键值对
                del temp_job['node_id']
            # 这段代码的作用——从temp_job中提取时间信息，只有在最近reward_interval时间内更新的作业信息才会被处理

            time_list = [int(x) for x in temp_job]
            if len(time_list) == 0:
                continue
            last_time = max(time_list)
            first_time = min(time_list)
            # 要求last_time和first_time都不为0
            if not last_time or not first_time:
                continue

            if j - last_time > reward_interval:  # reward_interval=1
                continue
            # 同样在限制处理的时间范围
            if j - first_time > reward_interval and str(j - reward_interval) in temp_job:  # ？？？？
                first_time = j - reward_interval  # j-1

            # 作业开始和结束的时间戳
            start_epoch_num = temp_job[str(first_time)]
            end_epoch_num = temp_job[str(last_time)]

            if start_epoch_num and end_epoch_num and last_time > first_time and end_epoch_num > start_epoch_num:
                running_num += 1
                if str(j - reward_interval) in job_epoch_records[job_id]:
                    state = job_epoch_records[job_id][str(j - reward_interval)]  # state是上一个时刻的状态信息
                    # TODO
                    real_value = (last_time - first_time) / (
                                end_epoch_num - start_epoch_num)  # ???为什么这样计算real_value 一个epoch的时间？？
                    # 将实际执行时间添加到性能分析中
                    performance.profiling_append(job_id, real_value)
                    # 根据状态预测的执行时间
                    predict_value = performance.make_predict(state)
                    # real_list、predict_list、loss_list、tuned_loss_list是存储不同指标的列表
                    real_list.append(real_value / performance.profiling_fetch(job_id))
                    predict_list.append(predict_value / performance.profiling_fetch(job_id))
                    loss_list.append((abs(predict_value - real_value) / real_value))
                    tuned_loss_list.append(
                        (max(abs(predict_value - real_value) - 2, 0.0001) / real_value))  # 经过调整的损失值，调整绝对误差
                    performance.update_samples(state, real_value)
                    del job_epoch_records[job_id][str(j - reward_interval)]  # 删除表示这个信息已经处理过了
                epoch += end_epoch_num - start_epoch_num

        # 这段代码负责调度算法的执行、奖励计算、策略选择、日志记录
        # 奖励函数的实现
        if len(tuned_loss_list) and len(predict_list):
            coe = sum(tuned_loss_list) / len(tuned_loss_list)  # tuned_loss_list的平均值，奖励函数的一个系数
            running_jobs[main_writer.get_step()]['tuned_loss'] = coe
            running_jobs[main_writer.get_step()]['real_loss'] = sum(loss_list) / len(loss_list)  # 计算loss_list列表的平均值
            # 计算奖励
            reward = np.clip(((1 - coe) * (len(predict_list) / sum(predict_list) * running_num) + coe * coe), -10, 10)
        else:  # 两个列表至少有一个为空
            if running_num != 0:
                if len(real_list):
                    reward = np.clip(len(real_list) / sum(real_list) * running_num, -10, 10)
                else:
                    reward = np.clip(running_num / min(len(job_name_list) - finish_cnt + 1, 5), -10, 10)
            else:
                reward = -10  # 设定一个初始的奖励值
        # 写入性能指标
        if real_list and loss_list and tuned_loss_list:
            main_writer.write_pm(np.mean(np.array(real_list)), np.mean(np.array(loss_list)),
                                 np.mean(np.array(tuned_loss_list)))
        running_jobs[main_writer.get_step()]['reward'] = reward
        main_writer.write_reward(reward)
        # 将运行作业的数量、已完成作业的数量、未完成作业的数量写入
        main_writer.write_job(running_num, finish_cnt, len(job_name_list) - finish_cnt - running_num)

        # environment reward
        # if running_num != 0:
        #     reward = min(sum(real_list) / len(real_list)+running_num/min(len(job_name_list)-finish_cnt+1, 5), 10)
        # else:
        #     reward
        # policy = schedulingAlgo.HorusScheduling(node_info, job_info, curr_time, 3, 1, 1)
        # policy = schedulingAlgo.DRFScheduling(node_info, job_info, curr_time)
        # policy = schedulingAlgo.averageScheduling(node_info, job_info)
        # policy = schedulingAlgo.cond2Scheduling(node_info, job_info)
        # policy = schedulingAlgo.cond3Scheduling(node_info, job_info)
        policy = agent.make_decision(node_info, job_info, node_dict, curr_time, reward, j)
        # print(agent.state_extraction(node_info, job_info, curr_time))
        # policy = RLScheduling(node_info, job_info, curr_time)
        print(policy)
        if policy:
            mylog.logger.info(f"in the {j} iteration the policy is {policy}")
            podScheduling(k8sCoreV1api, policy, job_tracker, job_epoch_records)
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        temp_log = OrderedDict()
        temp_log['node_info'] = node_info
        temp_log['job_info'] = job_info
        temp_log['policy'] = policy
        temp_log['step'] = main_writer.get_step()
        # log是run的参数
        log[log_time] = temp_log  # 将temp_log这个临时字典存储在log字典中，以当前的时间为键
        # main_writer.write_log(temp_log)
        main_writer.add_step()  # 增加步数
        print(log_time)
        td = (datetime.datetime.strptime(log_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(curr_time,
                                                                                                     '%Y-%m-%d %H:%M:%S')).seconds
        # if td < 180:
        #     time.sleep(180 - td)
        time.sleep(15)

        # print("{} log saved!!".format(log_path))
        # 根据已完成作业数量，判断是否该结束调度算法的循环， （作业完成后多久无法获取pod？）

        # 所有作业已完成，输出每个作业的完成时间，将日志数据写入文件，删除所有的提交作业，终止调度算法的循环
        if finish_cnt == len(job_name_list):  # 算法循环结束的条件：所有作业已完成
            for job in execution_time:
                print('Job {} : makespan is {} \n'.format(job, execution_time[job]))
            json_str = json.dumps(log, indent=4)  # 将字典对象转换为json格式的字符串，indent指定缩进为4个空格
            with open(log_path[0], 'w') as path:
                path.write(json_str)  # 将json字符串写入文件
            json_str1 = json.dumps(running_jobs, indent=4)
            with open(log_path[1], 'w') as path:
                path.write(json_str1)
            submission.delete_all(0)  # 删除所有提交的作业
            break
        finish_cnt = 0


def main(share_info):
    agent = RLschedulingAlgo.PPO(dim_dict, node_dict, hostinfo)
    performance = Performance('./model.ckpt', './jst-train_pkl', dim_dict)  # 似乎与模型文件和训练数据有关
    main_writer = monitor_writer()  # main_writer是monitor_writer类的一个实例，get_step是该类的一个方法，该方法返回main_writer的成员变量step

    # time.sleep(10)
    # time.sleep(100)

    for i in range(10):
        # 查询可用的节点信息
        useNodeName = getUseNode(k8sCoreV1api)
        # 这个random是 import numpy.random as random
        # replace设置为False表示一个元素不允许被重复选择
        job_name_list = random.choice(job_name_constants.job_name_list, 30, replace=False)
        random.shuffle(job_name_list)

        # 自动生成作业的yaml文件
        generate(job_name_list.tolist())

        # 将作业分组提交
        # 生成服从指数分布的随机数 scale表示指数分布的尺度参数(分布的平均值) size表示生成随机数的形状，如果size是单一整数，生成一个相应大小的数组
        sample_poisson = random.exponential(scale=15, size=len(job_name_list))  # 返回ndarray类型（NumPy的核心数据结构，表示多维数组）
        sample_poisson = [int(x) for x in sample_poisson]
        # 将原始列表前五分之一的元素用0替代 条件：i < 0.2* len(sample_poisson)
        # 并不是0的个数就是五分之一 原始列表中未被替换的元素也可能是0
        # 列表中0元素的个数大于等于五分之一
        sample_poisson = [0 if i < 0.2 * len(sample_poisson) else sample_poisson[i] for i in range(len(sample_poisson))]
        sample_poisson = sorted(sample_poisson)  # 默认升序
        # 该字典的目的是按sample_poisson中的随机值将作业名称进行分组
        submission_dict = OrderedDict()
        # 循环遍历sample_poisson中的每个值
        for i in range(len(sample_poisson)):
            if submission_dict.__contains__(sample_poisson[i]):
                submission_dict[sample_poisson[i]].append(job_name_list[i])
                # submission_dict[0] 至少有6个作业
            else:
                # 创建一个新键 sample_poisson[i]
                submission_dict[sample_poisson[i]] = []
                submission_dict[sample_poisson[i]].append(job_name_list[i])

        # 将字典submission_dict以二进制形式保存到文件中
        # 使用上下文管理器确保在操作完成后自动关闭文件
        with open('cur_submission_dict.pkl', 'wb') as f:  # 'wb'表示写入二进制数据
            # pickle模块的dump函数将字典写入文件中
            pickle.dump(submission_dict, f)  # 序列化（将python对象转换为二进制数据）

        log = OrderedDict()
        execution_time = OrderedDict()
        # 创建job_tracker字典(OrderedDict:30) 记录每个作业关联的node和gpu
        job_tracker = OrderedDict()
        for job_id in job_name_list:  # job_id 形如 'dcgan-7-2-10-16'
            job_tracker[job_id] = {'gpu_id': [], 'node_id': []}

        hostinfo['10.170.23.190']['GPUmemory_util'] = [5000, 0]  # master
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_name = 'RL-test' + '-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.json'
        log_name2 = 'Running' + '-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.json'
        # output_dir = './log/'
        log_path = [os.path.join(output_dir, log_name), os.path.join(output_dir, log_name2)]  # 两个path
        # gpu_monitor是main函数的参数
        run(job_name_list, log_path, start_time, useNodeName, log, execution_time, job_tracker, share_info, agent,
            performance, main_writer, submission_dict)
        time.sleep(120)


# 启动GPU监视器，获取主机和作业的信息 一直在更新信息
def start_monitor(share_info):
    gpu_monitor = gpuMonitor(hostinfo)
    # share_info初始化是一个{}
    share_info['host'] = gpu_monitor.get_host_info()
    share_info['job'] = gpu_monitor.get_pod_gpu_info()

    # 无限循环，不断地执行

    while True:
        gpu_monitor.update()  # 值更新
        share_info['host'] = gpu_monitor.get_host_info()  # 值获取
        share_info['job'] = gpu_monitor.get_pod_gpu_info()  # 值获取
        time.sleep(5)  # 每5s更新一次GPU监视信息


# 启动和管理多个进程以执行不同的任务
if __name__ == '__main__':  # 确保代码只在主程序中执行，而不是在被导入为模块时执行

    # 设置log
    mylog.logger.info("begin")
    # 设置多进程启动方式为spawn
    # torch.multiprocessing.set_start_method('spawn')
    manager = Manager()
    share_info = manager.dict()  # 这里是个空字典{}
    p1 = multiprocessing.Process(target=start_monitor, args=(share_info,), name="job_monitor")  # 监视GPU
    p1.daemon = True  # 将p1设置为守护进程，当主进程退出时，守护进程会自动终止
    p1.start()
    # time.sleep(10) # 让主进程休眠300s，等待p1进程执行一段时间后再启动p2
    main(share_info)
    gc.collect()
