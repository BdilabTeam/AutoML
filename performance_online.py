import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
import datetime
import pickle
import sa_dataloader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from zfilter import RunningStat
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# !!! 参数要改
input_size = 555
hidden_size = 128
num_classes = 1

batch_size = 1
learning_rate = 0.00001


node_dict = {'master': 1, 'node1': 2, 'node3': 3}
#
# N 为正在运行的作业的最大数量
N = 20
# L 为模型类别数量
L = 8
# K为单个节点上最大的GPU数量
K = 4
# M为节点总数
M = 3
# P为一个显卡上可以运行的最大worker总数
P = 8


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = F.tanh(out)
        out = self.fc2(out)
        return out

# def standardization(data):
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0) + 0.000001
#     return (data-mu)/sigma


def add_time_interval(state, time_interval):
    state = np.append(state, time_interval)
    # state = standardization(new_state)
    return state


class Performance:
    def __init__(self, model_path, sample_path, dim_dict):
        """
        :param model_path: 模型存储路径
        :param sample_path: 训练数据存储路径
        """
        self.dim_dict = dim_dict
        self.running_state = RunningStat(input_size)
        self.model_path = model_path
        self.model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        # self.model.load_state_dict(torch.load(model_path))
        self.sample_path = sample_path
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_GPU_num = self.dim_dict['max_GPU_num']
        self.max_node_num = self.dim_dict['max_node_num']
        self.max_worker_num = self.dim_dict['max_worker_num']
        self.max_job_type = self.dim_dict['max_job_type']
        self.max_job_running = self.dim_dict['max_job_running']
        self.max_job_gpu = self.dim_dict['max_job_gpu']
        self.batch_size = 1

        self.profiling_weight = dict()
        self.train_dataset = sa_dataloader.SaDataset(self.sample_path)
        self.train_writer = SummaryWriter(comment='train_pm')
        self.train_cnt = 0

    def profiling_append(self, job_id, real_value):
        job_meta = job_id.split('-')
        del job_meta[3]
        job_key = '-'.join(job_meta)
        if job_key in self.profiling_weight:
            self.profiling_weight[job_key].append(real_value)
        else:
            self.profiling_weight[job_key] = []
            self.profiling_weight[job_key].append(real_value)

    def profiling_fetch(self, job_id):
        job_meta = job_id.split('-')
        del job_meta[3]
        job_key = '-'.join(job_meta)
        if job_key in self.profiling_weight:
            return np.mean(np.array(self.profiling_weight[job_key]))
        else:
            avg = []
            for job_key in self.profiling_weight:
                avg.append(np.mean(np.array(self.profiling_weight[job_key])))
            return np.mean(avg)



    def state_extraction(self, node_info, job_info, target_job_id, job_resource_demand):
        """
        :param node_info:
        :param job_info:
        :param target_job_id: 待预测的job id
        :param job_resource_demand:
        :return: 不能预测则返回None，可以预测则返回state（即：x）
        """
        # 如果job没有开始运行或者已经结束则不预测，以及其他不能预测的情况
        if any(phase != 'Running' for phase in job_info[target_job_id]['phase']) or \
                'epoch' not in job_info[target_job_id] or job_info[target_job_id]['epoch'] is None or len(
                job_info[target_job_id]['epoch']) == 0 or job_info[target_job_id]['epoch'][0] is None:
            return None
        # v 节点上的显卡资源情况 M*4*K
        v = []
        # w 各节点上所有worker的作业类型，和请求
        w = []
        for node in node_info:
            filled_gpu_num = self.max_GPU_num - node_info[node]['GPU_num']
            gpu_utils = np.pad(np.array(node_info[node]['GPU_util']), (0, filled_gpu_num), 'constant',
                               constant_values=0)
            gpu_mem_util = np.pad(np.array(node_info[node]['GPUmemory_util']), (0, filled_gpu_num), 'constant',
                                  constant_values=0)
            gpu_mem_total = np.pad(np.array(node_info[node]['GPUmemory_total']), (0, filled_gpu_num), 'constant',
                                   constant_values=0)
            if node == 'master':
                gpu_type = np.array([1, 1, 1, 1])
            else:
                if node == 'node1':
                    gpu_type = np.array([2, 0, 0, 0])
                else:
                    gpu_type = np.array([3, 3, 0, 0])
            v.append(np.concatenate((gpu_utils, gpu_mem_util, gpu_mem_total, gpu_type)))

            worker_on_gpu = []
            for i in range(self.max_GPU_num):
                worker_on_gpu.append([])
            for job_id in job_info:
                # print(job_id)
                temp_job = job_info[job_id]
                job_type = temp_job['job_type']
                # batch_size = temp_job['batch_size']
                # GPUutil_demand = temp_job['GPUutil_demand']
                # GPUmem_demand = temp_job['GPUmem_demand']
                if all(phase == 'Running' for phase in temp_job['phase']):
                    for i in range(temp_job['worker_num']):
                        if temp_job['running_node'][i] == node_dict[node] and temp_job['gpu_id'][i] is not None and \
                                len(worker_on_gpu[temp_job['gpu_id'][i]]) < self.max_job_gpu * self.max_job_type:
                            worker_on_gpu[temp_job['gpu_id'][i]].extend(list(np.eye(self.max_job_type, dtype=np.float32)[job_type]))

            for i in range(self.max_GPU_num):
                if len(worker_on_gpu[i]) < self.max_job_gpu * self.max_job_type:
                    worker_on_gpu[i].extend(list(np.zeros(self.max_job_gpu * self.max_job_type-len(worker_on_gpu[i]), dtype=np.float32)))
            w.append(np.array(worker_on_gpu))
        v = np.array(v)
        w = np.array(w)


        job_type = (np.eye(L)[job_info[target_job_id]['job_type']])
        job_feature = np.array([job_info[target_job_id]['GPUmem_demand'], job_info[target_job_id]['worker_num'],
                                int(job_info[target_job_id]['batch_size'])])

        state = np.concatenate((np.reshape(v, -1), np.reshape(w, -1), job_type, job_feature))
        # print(np.shape(state))
        self.running_state.push(state)

        state = (state-self.running_state.mean) / (self.running_state.std + 0.0000001)

        return state

    def update_samples(self, state, true_epoch):
        """
        更新训练数据并保存
        :param state:
        :param true_epoch: 真实值
        :return:
        """
        self.train_dataset.add_sample(state, true_epoch)



    def update_model(self, num_epochs, save=True):
        """
        更新模型
        :param state:
        :param true_epoch: 真实值
        :param save: 是否保存模型
        :return:
        """
        self.model.train()
        pm_loss_list = []

        test_batch = BatchSampler(SubsetRandomSampler(range(len(self.train_dataset))), self.batch_size, True)
        index_iter = iter(test_batch)

        for epoch in range(num_epochs):
            for index in index_iter:
                # Move tensors to the configured device
                x, y = self.train_dataset[index]
                x = torch.from_numpy(x).float().to(device)
                x = x.view(self.batch_size, -1)
                y = torch.from_numpy(y).float().to(device)
                y = y.view(self.batch_size, -1)
                outputs = self.model(x)
                loss = self.criterion(outputs.view(-1), y)
                pm_loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if pm_loss_list:
            self.train_writer.add_scalar('train_pm/loss', np.mean(np.array(pm_loss_list)), global_step=self.train_cnt)
        self.train_cnt += 1
        if save:
            torch.save(self.model.state_dict(), self.model_path)
            self.train_dataset.sample_save()

    def make_predict(self, state):
        """
        :param state:
        :return: 不能预测则返回None，否则返回预测的epoch num
        """

        if state is not None:
            self.model.eval()
            x = torch.tensor(state, dtype=torch.float32).to(device)
            x = x.view(1, -1)

            predicted = self.model(x)
            return predicted.item()
        else:
            return None
