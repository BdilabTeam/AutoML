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

# N 为考虑的作业数量之和，包括已调度的和未调度的
N = 12
# L 为模型类别数量
L = 6
# K为考虑资源的总数，这里我们选择GPU利用率和GPU存储
K = 2
# M为节点总数
M = 2


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0) + 0.000001
    return (data-mu)/sigma


class Policy(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(dim_state, 32)

        self.action_head = nn.Linear(32, dim_action)
        self.value_head = nn.Linear(32, 1) # Scalar Value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value


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
            # v 节点上空闲的资源容量 M*K
            v = []
            for node_name in temp_node_info:
                GPU_util = np.mean(np.array(temp_node_info[node_name]['GPU_util']))
                GPUmemory_util = np.mean(np.array(temp_node_info[node_name]['GPUmemory_util']))
                GPUmemory_total = np.mean(np.array(temp_node_info[node_name]['GPUmemory_total']))
                v.append([1-GPU_util, GPUmemory_total-GPUmemory_util])
            v = np.array(v)

            # 生成与作业相关的state
            temp_job_info = log[tm]['job_info']
            job_num = len(temp_job_info)
            x = []
            r = []
            w = []
            d = np.zeros((M, N))
            for job_id in temp_job_info:
            # x 所属模型one-hot N*L
                x.append(temp_job_info[job_id]['job_type'])

            # r worker的资源请求 N*K
                r.append([temp_job_info[job_id]['GPUutil_demand'], temp_job_info[job_id]['GPUmem_demand']])
            # w 已分配的worker数量 N
                w.append(temp_job_info[job_id]['worker_num']-temp_job_info[job_id]['phase'].count('Pending'))
            # d 目前节点上运行worker的情况 M*N
                for node in temp_job_info[job_id]['running_node']:
                    if node:
                        d[node][x[-1]] += 1

            # x 转换为onehot
            x = np.eye(L)[x]
            x = np.pad(x, ((0, N - len(x)), (0, 0)), 'constant', constant_values=(0, 0))
            # r和w用0填充
            r = np.array(r)
            r = np.pad(r, ((0, N-len(r)), (0, 0)), 'constant', constant_values=(0, 0))
            w = np.array(w)
            w = np.pad(w, (0, N-len(w)), 'constant', constant_values=0)

            action = []
            state = np.concatenate(
                (np.reshape(v, -1), np.reshape(x, -1), np.reshape(r, -1), np.reshape(d, -1), np.reshape(w, -1)))
            # 将policy处理为action, action的定义为将第n个job的一个worker调度到第m个节点上,总大小为m*n
            temp_policy = log[tm]['policy']
            if temp_policy:
                for pod in temp_policy:
                    action_num = [(temp_policy[pod]['job_index']) * temp_policy[pod]['node_id']]
                    action = np.eye(N*M+1)[action_num]
                    # print(action)
                    action_list.append(np.reshape(action, -1))
                    state_list.append(state)
            else:
                action = np.reshape(np.eye(N*M+1)[0], -1)
                action_list.append(action)
                state_list.append(state)
    x = standardization(np.array(state_list))
    y = np.array(action_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    print(len(x_train))
    print(len(x_test))
    with open('offline-sa-train_pkl', 'wb') as f:
        pickle.dump([x_train, y_train], f)
    with open('offline-sa-test_pkl', 'wb') as f:
        pickle.dump([x_test, y_test], f)
