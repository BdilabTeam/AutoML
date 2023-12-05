import random
import numpy as np
import datetime

# 节点资源特征
# node_info[node_name]['CPU_core']
# node_info[node_name]['CPU_usage']
# node_info[node_name]['memory_size']
# node_info[node_name]['memory_usage']
# node_info[node_name]['GPU_util']
# node_info[node_name]['GPUmemory_util']
# node_info[node_name]['GPUmemory_total']


def capacity_check(node_info, job_info):
    spare_GPU = 0
    for node_name in node_info:
        temp_node_info = node_info[node_name]
        # CPU_indicator = temp_node_info['CPU_usage'] / temp_node_info['CPU_core'] < 1
        # mem_indicator = temp_node_info['memory_usage'] / temp_node_info['memory_size'] < 1
            # if CPU_indicator and mem_indicator:
        GPU_util = temp_node_info['GPU_util']
        GPUmemory_util = temp_node_info['GPUmemory_util']
        GPUmemory_total = temp_node_info['GPUmemory_total']
        for i in range(temp_node_info['GPU_num']):
            if (GPUmemory_util[i] + job_info['GPUmem_demand']) / GPUmemory_total[i] < 0.9:
                spare_GPU += 1
    if spare_GPU < job_info['worker_num']:
        return False
    else:
        return True


def randomScheduling(node_info, job_info, curr_time):

    policy = dict()
    for job_id in job_info:
        temp_job_info = job_info[job_id]
        if 'Pending' not in temp_job_info['phase']:
            continue
        if capacity_check(node_info, job_info[job_id]):
            worker_num = job_info[job_id]['worker_num']
            pod_list = job_info[job_id]['pod_list']
            namespace = job_info[job_id]['namespace']
            for i in range(worker_num):
                temp_policy = dict()
                temp_policy['job_id'] = job_id
                temp_policy['job_index'] = job_info[job_id]['job_index']
                temp_policy['namespace'] = namespace[i]
                temp_policy['node_name'] = random.choice(node_info.keys())
                temp_policy['node_id'] = node_info[temp_policy['node_name']]['node_id']
                temp_policy['gpu_index'] = None
                policy[pod_list[i]] = temp_policy
        return policy


# DRF 中主要是让用户公平使用集群资源，在我们的场景中，
# Min-max 算法需要知道作业在机器上的最早完成时间
# Horus算法中设k=1,即仅有一个队列

def HorusScheduling(node_info, job_info, curr_time, beta, omega1, omega2):
    GPU_info = dict()
    for node_name in node_info:
        temp = dict()
        temp['GPU_num'] = node_info[node_name]['GPU_num']
        temp['GPU_util'] = np.array(node_info[node_name]['GPU_util'])
        temp['GPUmemory_util'] = np.array(node_info[node_name]['GPUmemory_util'])
        temp['GPUmemory_total'] = np.array(node_info[node_name]['GPUmemory_total'])
        temp['GPU_worker'] = list(np.zeros_like(temp['GPU_util']))
        GPU_info[node_name] = temp
    policy = dict()
    # 首先选择beta个等待时间最长的job作为候选
    id_queue_time = []

    for job_id in job_info:
        temp_job_info = job_info[job_id]
        if 'Pending' not in temp_job_info['phase']:
            continue
        waiting_time = []
        creation_times = temp_job_info['creation_time']
        for creation_time in creation_times:
            waiting_time.append((datetime.datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S')-datetime.datetime.strptime(creation_time,'%Y-%m-%d %H:%M:%S')).seconds)
            # waiting_time.append(3)
        id_queue_time.append([job_id, np.mean(waiting_time)])
    id_queue_time.sort(key=lambda tup: tup[1], reverse=True)

    # print(id_queue_time)
    chosen_job_info = dict()
    for i in range(min(beta, len(id_queue_time))):
        job_id = id_queue_time[i][0]
        chosen_job_info[job_id] = job_info[job_id]
    # print(chosen_job_info)
    # 对于每一个job判断集群中是否有足够资源
    for job_id in chosen_job_info:
        if capacity_check(GPU_info, chosen_job_info[job_id]):
            GPUutil_demand = chosen_job_info[job_id]['GPUutil_demand']
            GPUmem_demand = chosen_job_info[job_id]['GPUmem_demand']
            GPU_cost = []
            for node_name in GPU_info:
                temp_node_info = node_info[node_name]
                for i in range(temp_node_info['GPU_num']):
                    temp_GPU_cost = dict()
                    temp_GPU_cost['mem_cost'] = (temp_node_info['GPUmemory_util'][i] + GPUmem_demand)/ \
                                                temp_node_info['GPUmemory_total'][i]
                    estimated_util = temp_node_info['GPU_util'][i] + GPUutil_demand
                    if estimated_util > 1:
                        temp_GPU_cost['util_cost'] = 1.2 * estimated_util
                    else:
                        temp_GPU_cost['util_cost'] = 0.8 * estimated_util
                    temp_GPU_cost['total_cost'] = omega1 * temp_GPU_cost['mem_cost'] + \
                                                  omega2 * temp_GPU_cost['util_cost']
                    temp_GPU_cost['gpu_index'] = i
                    temp_GPU_cost['node_name'] = node_name
                    temp_GPU_cost['node_id'] = temp_node_info['node_id']
                    GPU_cost.append(temp_GPU_cost)
            GPU_cost.sort(key=lambda dic: dic['total_cost'], reverse=False)
            worker_num = chosen_job_info[job_id]['worker_num']
            pod_list = chosen_job_info[job_id]['pod_list']
            if len(pod_list) != worker_num:
                return None
            for i in range(len(pod_list)):
                temp_policy = dict()
                temp_policy['job_id'] = job_id
                temp_policy['job_index'] = chosen_job_info[job_id]['job_index']
                temp_policy['namespace'] = chosen_job_info[job_id]['namespace']
                temp_policy['node_name'] = GPU_cost[i]['node_name']
                temp_policy['node_id'] = GPU_cost[i]['node_id']
                temp_policy['gpu_index'] = GPU_cost[i]['gpu_index']
                GPU_info[temp_policy['node_name']]['GPUmemory_util'][GPU_cost[i]['gpu_index']] += chosen_job_info[job_id]['GPUmem_demand']
                if GPU_info[temp_policy['node_name']]['GPUmemory_util'][GPU_cost[i]['gpu_index']] > 0.9 * GPU_info[temp_policy['node_name']]['GPUmemory_total'][GPU_cost[i]['gpu_index']]:

                    return None
                # print(GPU_info[temp_policy['node_name']]['GPUmemory_util'][GPU_cost[i]['gpu_index']])
                # print(GPU_info[temp_policy['node_name']]['GPUmemory_total'][GPU_cost[i]['gpu_index']])
                policy[pod_list[i]] = temp_policy
            # print(policy)
            return policy

def averageScheduling(node_info, job_info):

    policy = dict()
    for job_id in job_info:
        temp_job_info = job_info[job_id]
        if 'Pending' not in temp_job_info['phase']:
            continue
        # print(capacity_check(node_info, job_info[job_id]))
        if capacity_check(node_info, job_info[job_id]):
            worker_num = job_info[job_id]['worker_num']
            pod_list = job_info[job_id]['pod_list']
            namespace = job_info[job_id]['namespace']
            for i in range(worker_num):
                temp_policy = dict()
                temp_policy['job_id'] = job_id
                temp_policy['job_index'] = job_info[job_id]['job_index']
                temp_policy['namespace'] = namespace
                if i == 0:
                    temp_policy['node_name'] = 'master'
                else:
                    temp_policy['node_name'] = 'node1'
                temp_policy['node_id'] = i
                if i == 0:
                    temp_policy['gpu_index'] = 2
                else:
                    temp_policy['gpu_index'] = 0
                policy[pod_list[i]] = temp_policy
            return policy

def cond2Scheduling(node_info, job_info):

    policy = dict()
    for job_id in job_info:
        temp_job_info = job_info[job_id]
        if 'Pending' not in temp_job_info['phase']:
            continue
        if capacity_check(node_info, job_info[job_id]):
            worker_num = job_info[job_id]['worker_num']
            pod_list = job_info[job_id]['pod_list']
            namespace = job_info[job_id]['namespace']
            for i in range(worker_num):
                temp_policy = dict()
                temp_policy['job_id'] = job_id
                temp_policy['job_index'] = job_info[job_id]['job_index']
                temp_policy['namespace'] = namespace
                if i == 0:
                    temp_policy['node_name'] = 'master'
                else:
                    temp_policy['node_name'] = 'master'
                temp_policy['node_id'] = 0
                temp_policy['gpu_index'] = i+1
                policy[pod_list[i]] = temp_policy
            return policy

def cond3Scheduling(node_info, job_info):

    policy = dict()
    for job_id in job_info:
        temp_job_info = job_info[job_id]
        if 'Pending' not in temp_job_info['phase']:
            continue
        if capacity_check(node_info, job_info[job_id]):
            worker_num = job_info[job_id]['worker_num']
            pod_list = job_info[job_id]['pod_list']
            namespace = job_info[job_id]['namespace']
            for i in range(worker_num):
                temp_policy = dict()
                temp_policy['job_id'] = job_id
                temp_policy['job_index'] = job_info[job_id]['job_index']
                temp_policy['namespace'] = namespace
                if i == 0:
                    temp_policy['node_name'] = 'master'
                else:
                    temp_policy['node_name'] = 'master'
                temp_policy['node_id'] = 0
                temp_policy['gpu_index'] = 1
                policy[pod_list[i]] = temp_policy
            return policy





def someScheduling(node_info, job_info, curr_time):
    # 如何计算作业等待时间
    # delta = pod_info['creation_time'] - curr_time
    # print(delta.seconds)  # 间隔几秒
    policy = dict()
    # 调度策略
    # policy[pod_name] = dict()
    # policy[pod_name]['namespace'] = job_info[podName]['namespace']
    # policy[pod_name]['node_name'] = random.sample(node_resource.keys(), 1)

    # 节点资源特征
    # node_info[node_name]['CPU_core']
    # node_info[node_name]['CPU_usage']
    # node_info[node_name]['memory_size']
    # node_info[node_name]['memory_usage']
    # node_info[node_name]['GPU_num']
    # node_info[node_name]['GPU_util']
    # node_info[node_name]['GPUmemory_util']
    # node_info[node_name]['GPUmemory_total']

    # # pod特征
    # job_info[job_id]['namespace']
    # job_info[job_id]['pod_list']
    # job_info[job_id]['phase'] 包括Succeeded, Running, Pending三种
    # # pod 的资源请求
    # job_info[job_id]['CPU_demand']
    # job_info[job_id]['memory_demand']
    # job_info[job_id]['GPUutil_demand']
    # job_info[job_id]['GPUmem_demand']
    # # pod 所属job的信息
    # job_info[job_id]['job_id']
    # job_info[job_id]['job_type']
    # job_info[job_id]['worker_num']
    # # pod运行信息
    # job_info[job_id]['start_time'] #开始时间
    # job_info[job_id]['end_time']
    # job_info[job_id]['CPU_usage']
    # job_info[job_id]['memory_usage']
    # job_info[job_id]['creation_time'] #提交时间


    # 首先筛选出需要调度的job，判断是否job中有处于pending状态的worker, 然后将所有worker分配到master节点
    for job_id in job_info:
        temp_job_info = job_info[job_id]
        if 'Pending' not in temp_job_info['phase']:
            continue
        for i in range(len(temp_job_info['pod_list'])):
            if temp_job_info['phase'][i] == 'Pending':
                temp_policy = dict()
                temp_policy['namespace'] = temp_job_info['namespace'][i]
                temp_policy['node_name'] = 'master'
                policy[temp_job_info['pod_list'][i]] = temp_policy
    return policy