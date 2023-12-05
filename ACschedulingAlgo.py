import random
import numpy as np
import datetime
import gym
from collections import namedtuple, OrderedDict
from itertools import count
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

import os

eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'reward'])
gamma = 0.99
render = False
seed = 1
log_interval = 10


class Policy(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(dim_state, 32)

        self.action_head = nn.Linear(32, dim_action)
        self.value_head = nn.Linear(32, 1) # Scalar Value

        os.makedirs('./AC_CartPole-v0', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value





class AC():
    learning_rate = 0.01
    gamma = 0.99
    episodes = 20000
    ppo_update_time = 10
    buffer_capacity = 8000
    batch_size = 32

    def __init__(self, dim_dict, node_dict):

        super(AC, self).__init__()
        self.max_node_num = dim_dict['max_node_num']
        self.max_GPU_num = dim_dict['max_GPU_num']
        self.max_job_running = dim_dict['max_job_running']
        self.max_worker_num = dim_dict['max_worker_num']
        self.max_job_type = dim_dict['max_job_type']
        self.node_dict = node_dict
        # 判断之前调度的job是否已全部完成，应该开始下一个job的调度
        self.next_job_flag = True
        self.scheduling_job_dict = None
        self.last_action = -1
        self.temp_state_action = []

        self.reward_interval = 5

        self.dim_state = 158
        self.dim_action = self.max_node_num + 1

        self.model = Policy(self.dim_state, self.dim_action)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def state_extraction(self, node_info, job_info, curr_time):
        schedule_flag = False

        # 提取出节点信息中所需特征， 维度为 节点数量，（3* 最大GPU数量）
        node_feature = []
        for node in node_info:
            filled_gpu_num = self.max_GPU_num - node_info[node]['GPU_num']
            gpu_utils = np.pad(np.array(node_info[node]['GPU_util']), (0, filled_gpu_num), 'constant', constant_values=-1)
            gpu_mem_util = np.pad(np.array(node_info[node]['GPUmemory_util']), (0, filled_gpu_num), 'constant', constant_values=-1)
            gpu_mem_total = np.pad(np.array(node_info[node]['GPUmemory_total']), (0, filled_gpu_num), 'constant', constant_values=-1)
            node_feature.append(np.concatenate((gpu_utils, gpu_mem_util, gpu_mem_total)))
        node_feature = np.array(node_feature)

        # 提取出正在运行的worker特征， 按照所在节点整理
        worker_features = []
        x = []
        r = []

        w = []
        queued_job_score = OrderedDict()
        d = np.zeros((self.max_job_type, self.max_job_running))
        for job_id in job_info:
            if all([phase == 'Running' for phase in job_info[job_id]['phase']]):
                x.append(job_info[job_id]['job_type'])
                # r worker的资源请求 维度为 最大作业数量，job请求资源维数
                r.append([job_info[job_id]['GPUutil_demand'], job_info[job_id]['GPUmem_demand']])
                # w 已分配的worker数量 维度为 最大作业数量
                w.append(job_info[job_id]['worker_num'] - job_info[job_id]['phase'].count('Pending'))
                # d 目前节点上运行worker的情况 维度为 节点数量，最大作业数量
                for node in job_info[job_id]['running_node']:
                    if node:
                        d[node][x[job_info[job_id]['job_type']]] += 1
            else:
                if self.next_job_flag and all([phase == 'Pending' for phase in job_info[job_id]['phase']]):
                    queued_job_score[job_id] = (datetime.datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S')
                                                -datetime.datetime.strptime(job_info[job_id]['creation_time'][0], '%Y-%m-%d %H:%M:%S')).seconds

        # x 转换为onehot
        if not r:
            x.append(0)
        x = np.eye(self.max_job_type)[x]
        x = np.pad(x, ((0, self.max_job_running - len(x)), (0, 0)), 'constant', constant_values=(0, 0))
        # r和w用0填充
        if not r:
            r.append([0, 0])
        r = np.array(r)
        r = np.pad(r, ((0, self.max_job_running - len(r)), (0, 0)), 'constant', constant_values=(0, 0))
        w = np.array(w)
        w = np.pad(w, (0, self.max_job_running - len(w)), 'constant', constant_values=0)

        j = []
        if not self.scheduling_job_dict or self.scheduling_job_dict['unscheduled_worker_num'] == 0:
            self.next_job_flag = True
        if self.next_job_flag:
            queued_job_score = sorted(queued_job_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            if len(queued_job_score) > 0:
                schedule_flag = True
                scheduling_job_id = queued_job_score[0][0]
                self.scheduling_job_dict = {'pod_list': job_info[scheduling_job_id]['pod_list'],
                                            'job_type': job_info[scheduling_job_id]['job_type'],
                                            'job_id': scheduling_job_id,
                                            'namespace': job_info[scheduling_job_id]['namespace'],
                                            'worker_num': job_info[scheduling_job_id]['worker_num'],
                                            'unscheduled_worker_num': job_info[scheduling_job_id]['worker_num'],
                                            'GPUutil_demand': job_info[scheduling_job_id]['GPUutil_demand'],
                                            'GPUmem_demand': job_info[scheduling_job_id]['GPUmem_demand'],
                                            'scheduled_node': np.full(self.max_worker_num, fill_value=-1, dtype=int)}
            else:
                schedule_flag = False
                self.scheduling_job_dict = {'pod_list': [None],
                                            'job_type': 0,
                                            'job_id': None,
                                            'namespace': None,
                                            'worker_num': 0,
                                            'unscheduled_worker_num': 0,
                                            'GPUutil_demand': 0,
                                            'GPUmem_demand': 0,
                                            'scheduled_node': np.full(self.max_worker_num, fill_value=-0, dtype=int)}

        jx = np.eye(self.max_job_type)[self.scheduling_job_dict['job_type']]
        js = self.scheduling_job_dict['scheduled_node']
        j.append(self.scheduling_job_dict['GPUutil_demand'])
        j.append(self.scheduling_job_dict['GPUmem_demand'])
        j.append(self.scheduling_job_dict['worker_num'])
        j.append(self.scheduling_job_dict['unscheduled_worker_num'])

        state = np.concatenate(
            (np.reshape(node_feature, -1), np.reshape(x, -1), np.reshape(r, -1), np.reshape(w, -1), np.reshape(d, -1),
             np.reshape(np.array(j), -1), np.reshape(jx, -1), js),
        )
        # print(np.shape(node_feature))
        # print(np.shape(x))
        # print(np.shape(r))
        # print(np.shape(w))
        # print(np.shape(d))
        # print(np.shape(j))
        # print(np.shape(jx))
        # print(np.shape(js))
        # print(np.shape(state))
        return state, schedule_flag



    # 提取出等待调度的作业特征
    def make_decision(self, node_info, job_info, curr_time, reward, j):
        # 提取出各个节点上所有GPU的使用情况
        GPU_info = dict()
        for node_name in node_info:
            temp = dict()
            temp['GPU_num'] = node_info[node_name]['GPU_num']
            temp['GPU_util'] = np.array(node_info[node_name]['GPU_util'])
            temp['GPUmemory_util'] = np.array(node_info[node_name]['GPUmemory_util'])
            temp['GPUmemory_total'] = np.array(node_info[node_name]['GPUmemory_total'])
            GPU_info[node_name] = temp

        state, schedule_flag = self.state_extraction(node_info, job_info, curr_time)
        self.receive_reward(state, reward, j)
        action, probs, value, log_prob = self.select_action(state)
        if not schedule_flag:
            if action:
                trans = SavedAction(log_prob, value, -1)
            else:
                trans = SavedAction(log_prob, value, 0)
            self.store_transition(trans)
            return None
        policy = dict()
        for i in range(self.scheduling_job_dict['worker_num']):
            # print(self.scheduling_job_dict)
            curr_policy = dict()
            state, schedule_flag = self.state_extraction(node_info, job_info, curr_time)
            action, probs, value, log_prob = self.select_action(state)
            # print(probs)
            # action = 0
            # action_prob = [0.8, 0.2]
            self.temp_state_action.append({'probs': log_prob, 'value': value, 'step': i})

            if action == 0:
                trans = SavedAction(log_prob, value, 0)
                self.store_transition(trans)
                return None
            # 首先筛选显存要求足够的的GPU，选择一个最空闲的GPU
            curr_node = GPU_info[self.node_dict[str(int(action)-1)]]
            mem_cap = [i for i, x in enumerate(curr_node['GPUmemory_util']) if (
                        x + self.scheduling_job_dict['GPUmem_demand'] / self.scheduling_job_dict['worker_num'] <
                        curr_node['GPUmemory_total'][0])]

            min_util = 100
            min_gpu = -1
            for id in mem_cap:
                if curr_node['GPU_util'][id] < min_util:
                    min_gpu = id
            if len(mem_cap) == 0 or min_gpu == -1:
                trans = SavedAction(log_prob, value, -1)
                self.store_transition(trans)
                del self.temp_state_action[-1]
                return None

            self.scheduling_job_dict['scheduled_node'][self.scheduling_job_dict['worker_num']
                                                       - self.scheduling_job_dict['unscheduled_worker_num']] = int(action-1)
            self.scheduling_job_dict['unscheduled_worker_num'] -= 1

            curr_policy['namespace'] = self.scheduling_job_dict['namespace']
            curr_policy['node_name'] = self.node_dict[str(int(action)-1)]
            curr_policy['node_id'] = int(action-1)

            GPU_info[curr_policy['node_name']]['GPU_util'][min_gpu] += self.scheduling_job_dict['GPUutil_demand']
            GPU_info[curr_policy['node_name']]['GPUmemory_util'][min_gpu] += self.scheduling_job_dict['GPUmem_demand']
            curr_policy['gpu_index'] = min_gpu

            policy[self.scheduling_job_dict['pod_list'][i]] = curr_policy
        return policy

    def receive_reward(self, next_state, reward, j):
        for sa in self.temp_state_action:
            if j - sa['step'] > self.reward_interval:
                trans = SavedAction(sa['probs'], sa['value'], reward)
                self.store_transition(trans)
        self.temp_state_action = []
        return

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, value = self.model(state)
        c = Categorical(probs)
        action = c.sample()
        log_prob = c.log_prob(action)
        return action, probs, value, log_prob

    def save_param(self):
        torch.save(self.model.state_dict(), '../param/net_param/policy' + str(time.time())[:10] + '.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_epoch):
        log_prob = torch.tensor([t.log_prob for t in self.buffer], dtype=torch.float, requires_grad=True)
        value = torch.tensor([t.value for t in self.buffer], dtype=torch.float, requires_grad=True).view(-1, 1)
        rewards = [t.reward for t in self.buffer]
        R = 0
        policy_loss = []
        value_loss = []
        for r in rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, requires_grad=True)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_epoch, self.training_step))
                reward = rewards[index].view(-1) - value[index].view(-1)
                policy_loss.append(-log_prob[index] * reward)
                # print(F.smooth_l1_loss(value[index].view(-1), torch.tensor(rewards[index]).view(-1)))
                value_loss.append(F.smooth_l1_loss(value[index].view(-1), rewards[index].view(-1)))
            if value_loss:
                self.optimizer.zero_grad()
                loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.training_step += 1

        # del self.buffer[:]