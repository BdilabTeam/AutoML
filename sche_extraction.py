import json
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
from sklearn.model_selection import train_test_split

max_node_num = 2
max_GPU_num = 4
max_worker_num = 5
max_job_type = 8
max_job_running = 12


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0) + 0.000001
    return (data-mu)/sigma





if __name__ == '__main__':
    # 读入log文件

    experiment_name = 'Horus-test'
    time_stamps = ['20211104204818', '20211104232126', '20211105022141',
                   '20211105041705', '20211105070944', '20211105093602', '20211105122805', '20211105145457',
                   '20211105145457', '20211105175523', '20211105234228', '20211106021626', '20211106050206',
                   '20211106075027', '20211106105318', '20211106150439', '20211106175416', '20211106203203',
                   '20211106225659', '20211107021951', '20211107132718', '20211107163512', '20211107210905',
                   '20211108000626', '20211108025402', '20211108050902', '20211108074420', '20211108094545',
                   '20211108120056']
    # time_stamps = ['20211104204818', '20211104232126']
    state_list = []
    action_list = []
    paths = []
    for time_stamp in time_stamps:
        paths.append('logs/horus_logs/' + experiment_name + '-' + time_stamp + '/' + 'log.json')
    for log_path in paths:
        log = json.load(open(log_path), object_pairs_hook=OrderedDict)
        for tm in log:

            # 生成与节点相关的state
            temp_node_info = log[tm]['node_info']

            # 提取出节点信息中所需特征， 维度为 节点数量，（3* 最大GPU数量）
            node_feature = []
            for node in temp_node_info:
                filled_gpu_num = max_GPU_num - temp_node_info[node]['GPU_num']
                gpu_utils = np.pad(np.array(temp_node_info[node]['GPU_util'])/100, (0, filled_gpu_num), 'constant',
                                   constant_values=-1)
                gpu_mem_util = np.pad(np.array(temp_node_info[node]['GPUmemory_util'])/1000, (0, filled_gpu_num), 'constant',
                                     constant_values=-1)
                gpu_mem_total = np.pad(np.array(temp_node_info[node]['GPUmemory_total'])/1000, (0, filled_gpu_num), 'constant',
                                      constant_values=-1)
                node_feature.append(np.concatenate((gpu_utils, gpu_mem_util, gpu_mem_total)))
            node_feature = np.array(node_feature)

            # 提取出正在运行的worker特征， 按照所在节点整理
            worker_features = []
            x = []
            r1 = []
            r2 = []
            r3 = []
            r4 = []
            w = []
            d = np.zeros((max_node_num, max_job_running))
            temp_job_info = log[tm]['job_info']
            for job_id in temp_job_info:
                if all([phase == 'Running' for phase in temp_job_info[job_id]['phase']]):
                    x.append(temp_job_info[job_id]['job_type'])
                    # r worker的资源请求 维度为 最大作业数量，job请求资源维数, 剩余epoch数量
                    if temp_job_info[job_id]['epoch'] and temp_job_info[job_id]['epoch'][0]:
                        r1.append(temp_job_info[job_id]['batch_size'])
                        r2.append(temp_job_info[job_id]['GPUmem_demand']/1000)
                        r3.append((temp_job_info[job_id]['epoch_num'] - temp_job_info[job_id]['epoch'][0]) /
                                  temp_job_info[job_id]['epoch_num'])

                        r4.append(temp_job_info[job_id]['epoch_num']/10)


                    else:
                        r1.append(temp_job_info[job_id]['batch_size'])
                        r2.append(temp_job_info[job_id]['GPUmem_demand']/1000)
                        r3.append(1)
                        r4.append(temp_job_info[job_id]['epoch_num']/10)

                    # w 已分配的worker数量 维度为 最大作业数量
                    # w.append(temp_job_info[job_id]['worker_num'] - temp_job_info[job_id]['phase'].count('Pending'))
                    # d 目前节点上运行worker的情况 维度为 节点数量，最大作业数量
                    for node in temp_job_info[job_id]['running_node']:
                        if node:
                            d[node][len(x)] += 1

            # x 转换为onehot
            if not x:
                x.append(0)
            x = np.eye(max_job_type)[x]
            x = np.pad(x, ((0, max_job_running - len(x)), (0, 0)), 'constant', constant_values=(0, 0))
            # r和w用0填充
            r1 = np.pad(np.array(r1), (0, max_job_running - len(r1)), 'constant', constant_values=0)
            r2 = np.pad(np.array(r2), (0, max_job_running - len(r2)), 'constant', constant_values=0)
            r3 = np.pad(np.array(r3), (0, max_job_running - len(r3)), 'constant', constant_values=0)
            r4 = np.pad(np.array(r4), (0, max_job_running - len(r4)), 'constant', constant_values=0)
            # w = np.array(w)
            # w = np.pad(w, (0, max_job_running - len(w)), 'constant', constant_values=0)
            temp_policy = log[tm]['policy']
            policy_list = []
            scheduling_job_id = None
            scheduling_job_dict = dict()
            if temp_policy:
                for pod in temp_policy:
                    policy_list.append(temp_policy[pod]['node_id'])
                    scheduling_job_id = temp_policy[pod]['job_id']

            j = []
            if scheduling_job_id:
                schedule_flag = True
                scheduling_job_dict = {'pod_list': temp_job_info[scheduling_job_id]['pod_list'],
                                            'job_type': temp_job_info[scheduling_job_id]['job_type'],
                                            'job_id': scheduling_job_id,
                                            'namespace': temp_job_info[scheduling_job_id]['namespace'],
                                            'worker_num': temp_job_info[scheduling_job_id]['worker_num'],
                                            'unscheduled_worker_num': temp_job_info[scheduling_job_id]['worker_num'],
                                            'GPUutil_demand': temp_job_info[scheduling_job_id]['GPUutil_demand'],
                                            'GPUmem_demand': temp_job_info[scheduling_job_id]['GPUmem_demand'],
                                            'scheduled_node': np.full(max_worker_num, fill_value=-1,
                                                                      dtype=int),
                                            'batch_size': temp_job_info[scheduling_job_id]['batch_size']
                                       }
                for i in range(len(policy_list)):
                    j = []
                    jx = np.eye(max_job_type)[scheduling_job_dict['job_type']]
                    js = scheduling_job_dict['scheduled_node']
                    j.append(scheduling_job_dict['batch_size'])
                    j.append(scheduling_job_dict['GPUmem_demand'])
                    j.append(scheduling_job_dict['worker_num'])
                    j.append(scheduling_job_dict['unscheduled_worker_num'])

                    state = np.concatenate(
                        (np.reshape(node_feature, -1), np.reshape(x, -1), np.reshape(r1, -1),
                         np.reshape(r2, -1), np.reshape(r3, -1), np.reshape(r4, -1),
                         np.reshape(w, -1),
                         np.reshape(d, -1),
                         np.reshape(np.array(j), -1), np.reshape(jx, -1), js),
                    )

                    state_list.append(state)
                    action_list.append(np.eye(max_node_num+1)[policy_list[i]+1])
            else:
                schedule_flag = False
                scheduling_job_dict = {'pod_list': [None],
                                            'job_type': 0,
                                            'job_id': None,
                                            'namespace': None,
                                            'worker_num': 0,
                                            'unscheduled_worker_num': 0,
                                            'GPUutil_demand': 0,
                                            'GPUmem_demand': 0,
                                            'scheduled_node': np.full(max_worker_num, fill_value=-0,
                                                                      dtype=int),
                                       'batch_size': 0}
                j = []
                jx = np.eye(max_job_type)[scheduling_job_dict['job_type']]
                js = scheduling_job_dict['scheduled_node']
                j.append(scheduling_job_dict['batch_size'])
                j.append(scheduling_job_dict['GPUmem_demand'])
                j.append(scheduling_job_dict['worker_num'])
                j.append(scheduling_job_dict['unscheduled_worker_num'])
                state = np.concatenate(
                    (np.reshape(node_feature, -1), np.reshape(x, -1), np.reshape(r1, -1),
                     np.reshape(r2, -1), np.reshape(r3, -1), np.reshape(r4, -1),
                     np.reshape(w, -1),
                     np.reshape(d, -1),
                     np.reshape(np.array(j), -1), np.reshape(jx, -1), js),
                )

                # print(state.shape)
                state_list.append(state)
                action_list.append(np.eye(max_node_num+1)[0])


    # x = standardization(np.array(state_list))
    x = np.array(state_list)
    print(x.shape)
    y = np.array(action_list)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)
    print(len(x_train))
    print(len(x_test))
    with open('offline-sche-train_pkl', 'wb') as f:
        pickle.dump([x_train, y_train], f)
    with open('offline-sche-test_pkl', 'wb') as f:
        pickle.dump([x_test, y_test], f)
