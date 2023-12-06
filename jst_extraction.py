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


#
# N 为正在运行的作业的最大数量
N = 20
# L 为模型类别数量
L = 8
# K为单个节点上最大的GPU数量
K = 4
# M为节点总数
M = 4
# P为一个显卡上可以运行的最大worker总数
P = 8

state_list = []
action_list = []
node_dict = {'master': 1, 'node1': 2, 'node2': 3, 'node3': 4}
job_resource_demand = {'vgg-1-3-30-16': {'GPUmem_demand': 5000, 'GPUutil_demand': 90, 'time_per_epoch': 100},
                       'vgg-1-2-50-32': {'GPUmem_demand': 5000, 'GPUutil_demand': 90, 'time_per_epoch': 60},
                       'resnet-2-2-30-8': {'GPUmem_demand': 5000, 'GPUutil_demand': 30, 'time_per_epoch': 60},
                       'resnet-2-3-50-16': {'GPUmem_demand': 5000, 'GPUutil_demand': 35, 'time_per_epoch': 20},
                       'unet-3-2-30-8': {'GPUmem_demand': 4500, 'GPUutil_demand': 45, 'time_per_epoch': 100},
                       'unet-3-3-50-16': {'GPUmem_demand': 4500, 'GPUutil_demand': 60, 'time_per_epoch': 35},
                        'cnn-4-4-30-16': {'GPUmem_demand': 1100, 'GPUutil_demand': 30, 'time_per_epoch': 70},
                       'cnn-4-5-50-32': {'GPUmem_demand': 1100, 'GPUutil_demand': 35, 'time_per_epoch': 30},
                       'rnn-5-2-30-8': {'GPUmem_demand': 1300, 'GPUutil_demand': 40, 'time_per_epoch': 30},
                       'rnn-5-3-50-16': {'GPUmem_demand': 1300, 'GPUutil_demand': 50, 'time_per_epoch': 10},
                        'cnnl-6-2-30-16': {'GPUmem_demand': 1100, 'GPUutil_demand': 50, 'time_per_epoch': 25},
                       'cnnl-6-3-50-32': {'GPUmem_demand': 1100, 'GPUutil_demand': 60, 'time_per_epoch': 10},
                        'seq2seq': {'GPUmem_demand': 1100, 'GPUutil_demand': 60, 'time_per_epoch': 30},
                       }


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0) + 0.000001
    return (data-mu)/sigma

if __name__ == '__main__':
    # 读入log文件
    x = []
    y = []
    paths = []
    alloc_paths = []
    experiment_name = 'RL-test'
    time_stamps = ['20220614111858']
    for time_stamp in time_stamps:
        paths.append('./log/' + experiment_name + '-' + time_stamp + '/' + 'log.json')
        alloc_paths.append('./log/' + experiment_name + '-' + time_stamp + '/' + 'job_allocation.json')
    for log_path, alloc_path in zip(paths,alloc_paths):
        log = json.load(open(log_path), object_pairs_hook=OrderedDict)
        alloc = json.load(open(alloc_path), object_pairs_hook=OrderedDict)
        print(alloc)
        tms = list(log.keys())
        for j in range(len(tms)-1):
            tm = tms[j]
            # 生成与节点相关的state
            temp_node_info = log[tm]['node_info']
            temp_job_info = log[tm]['job_info']
            # v 节点上的显卡资源情况 M*4*K
            v = []
            # w 各节点上所有worker的作业类型，和请求
            w = []
            for node_name in temp_node_info:
                GPU_util = np.array(temp_node_info[node_name]['GPU_util'])
                GPU_util = np.pad(GPU_util, (0, K - len(GPU_util)), 'constant', constant_values=0)
                GPUmem_util = np.array(temp_node_info[node_name]['GPUmemory_util'])
                GPUmem_util = np.pad(GPUmem_util, (0, K - len(GPUmem_util)), 'constant', constant_values=0)
                GPUmem_total = np.array(temp_node_info[node_name]['GPUmemory_total'])
                GPUmem_total = np.pad(GPUmem_total, (0, K - len(GPUmem_total)), 'constant', constant_values=0)
                if node_name == 'master':
                    GPU_type = np.array([1, 1, 1, 1])
                else:
                    if node_name == 'node1':
                        GPU_type = np.array([2, 0, 0, 0])
                    else:
                        GPU_type = np.array([3, 3, 0, 0])
                v.append(np.concatenate((GPU_util, GPUmem_util, GPUmem_total, GPU_type)))

                worker_on_gpu = []
                for i in range(K):
                    worker_on_gpu.append([])
                for job_id in temp_job_info:
                    # print(job_id)
                    temp_job = temp_job_info[job_id]
                    job_type = temp_job['job_type']
                    batch_size = temp_job['batch_size']
                    GPUmem_demand = temp_job['GPUmem_demand']
                    temp_job['gpu_id'] = alloc[job_id]['gpu_index']
                    temp_job['running_node'] = alloc[job_id]['node_name']
                    if all(phase == 'Running' for phase in temp_job['phase']):
                        for i in range(temp_job['worker_num']):
                            if temp_job['running_node'][i] == node_name:
                                worker_on_gpu[temp_job['gpu_id'][i]].append(np.eye(L)[job_type])
                for i in range(K):
                    while len(worker_on_gpu[i]) < P:
                        worker_on_gpu[i].append(np.zeros(L))
                w.append(np.array(worker_on_gpu))
            v = np.array(v)
            w = np.array(w)
            print(v.shape)
            print(w.shape)

            # print(np.shape(v))
            # print(np.shape(w))
            # 生成job相关的slowdown
            for job_id in temp_job_info:
                if 'epoch' not in temp_job_info[job_id] or temp_job_info[job_id]['epoch'] is None or not len(temp_job_info[job_id]['epoch']):
                    break
                curr_epoch = temp_job_info[job_id]['epoch'][0]
                if not curr_epoch or curr_epoch < 0:
                    continue
                curr_epoch_time = temp_job_info[job_id]['epoch_time'][0]
                next_epoch = log[tms[j + 1]]['job_info'][job_id]['epoch'][0]
                next_epoch_time = log[tms[j + 1]]['job_info'][job_id]['epoch_time'][0]
                if not next_epoch or next_epoch < 0:
                    continue
                if not next_epoch_time or next_epoch_time == 0 or not curr_epoch_time or curr_epoch_time == 0:
                    continue
                job_type = (np.eye(L)[temp_job_info[job_id]['job_type']])
                job_feature = np.array([temp_job_info[job_id]['GPUmem_demand'], temp_job_info[job_id]['worker_num'],
                                        int(temp_job_info[job_id]['batch_size'])])
                if curr_epoch != 0 and next_epoch != curr_epoch:
                    x.append(np.concatenate((np.reshape(v, -1), np.reshape(w, -1), job_type, job_feature)))
                    y.append((next_epoch_time-curr_epoch_time)/(next_epoch-curr_epoch))
                    print((next_epoch_time-curr_epoch_time)/(next_epoch-curr_epoch))

                    # print(epoch_time)
                    # epoch_time += term / 60
                # slowdown += (second_epoch_time - first_epoch_time) / 60

                # print(slowdown)
            # if epoch_time != 0:
            #     x.append(np.concatenate((np.reshape(v, -1), np.reshape(w, -1))))
            #     # print(cnt)
            #     #print(epoch_time/cnt)
            #     y.append(epoch_time/cnt)

    # x = standardization(np.array(x))
    x= np.array(x)
    print(x.shape)
    y = np.array(y)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=777)
    print(len(x_train))
    print(len(x_test))
    with open('jst-train_pkl', 'wb') as f:
        pickle.dump([x_train, y_train], f)
    with open('jst-test_pkl', 'wb') as f:
        pickle.dump([x_test, y_test], f)



