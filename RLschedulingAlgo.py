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
import pickle
from zfilter import RunningStat

import os

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'step'])
gamma = 0.99
render = False
seed = 6666
log_interval = 10


class Actor(nn.Module):
    def __init__(self, num_state, num_action, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state, hidden_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(num_state, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 8000
    batch_size = 4

    def __init__(self, dim_dict, node_dict, host_info):
        super(PPO, self).__init__()
        self.max_node_num = dim_dict['max_node_num']
        self.max_GPU_num = dim_dict['max_GPU_num']
        self.max_job_running = dim_dict['max_job_running']
        self.max_worker_num = dim_dict['max_worker_num']
        self.max_job_type = dim_dict['max_job_type']
        self.max_job_gpu = dim_dict['max_job_gpu']
        self.node_dict = node_dict
        self.gpu_host_array = []
        self.gpu_host_array.append([-1, -1])
        for host in host_info:

            for i in range(host_info[host]['gpu_num']):
                self.gpu_host_array.append([host_info[host]['node_id'], i])

        # 判断之前调度的job是否已全部完成，应该开始下一个job的调度
        self.next_job_flag = True
        self.scheduling_job_dict = None
        self.last_action = -1
        self.temp_state_action = []

        self.reward_interval = 1

        self.dim_state = 562

        self.dim_action = len(self.gpu_host_array)
        self.runnin_state = RunningStat(self.dim_state)
        self.actor_net = Actor(self.dim_state, self.dim_action, 128).float()
        # self.actor_net.load_state_dict(torch.load('model32.ckpt'))
        self.critic_net = Critic(self.dim_state, 128).float()
        # self.load_param('1653183543', '1653183543')
        self.buffer = []
        # with open('replay_buffer_pkl', 'rb') as f:
        #      self.buffer = pickle.load(f)
        self.buffer_counter = 0
        self.cnt1 = 0
        self.cnt2 = 0
        self.training_cnt = 0
        self.train_writer = SummaryWriter(comment='train_agent')
        self.action_writer = SummaryWriter(comment='action')
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 3e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-4)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def state_extraction(self, node_info, job_info, node_dict, curr_time):
        schedule_flag = False

        # 提取出节点信息中所需特征， 维度为 节点数量，（3* 最大GPU数量）
        node_feature = []
        node_worker = []
        for node in node_info:
            filled_gpu_num = self.max_GPU_num - node_info[node]['GPU_num']
            gpu_utils = np.pad(np.array(node_info[node]['GPU_util']), (0, filled_gpu_num), 'constant',
                               constant_values=-1)
            gpu_mem_util = np.pad(np.array(node_info[node]['GPUmemory_util']), (0, filled_gpu_num), 'constant',
                                  constant_values=-1)
            gpu_mem_total = np.pad(np.array(node_info[node]['GPUmemory_total']), (0, filled_gpu_num), 'constant',
                                   constant_values=-1)
            node_feature.append(np.concatenate((gpu_utils, gpu_mem_util, gpu_mem_total)))
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
                        if temp_job['running_node'][i] == node and temp_job['gpu_id'][i] is not None and \
                                len(worker_on_gpu[temp_job['gpu_id'][i]]) < self.max_job_type * self.max_job_gpu:
                            # print(temp_job['gpu_id'][i], np.eye(L)[job_type], job_type)
                            worker_on_gpu[temp_job['gpu_id'][i]].extend(
                                list(np.eye(self.max_job_type, dtype=np.float)[job_type]))
            for i in range(self.max_GPU_num):
                if len(worker_on_gpu[i]) < self.max_job_type * self.max_job_gpu:
                    worker_on_gpu[i].extend(
                        list(np.zeros(self.max_job_type * self.max_job_gpu - len(worker_on_gpu[i]), dtype=np.float32)))
            node_worker.append(worker_on_gpu)
        node_feature = np.array(node_feature)
        node_worker = np.array(node_worker)

        # 提取出正在运行的worker特征， 按照所在节点整理
        x = []
        r1 = []
        r2 = []
        r3 = []
        r4 = []
        w = []
        queued_job_score = OrderedDict()
        d = np.zeros((self.max_node_num, self.max_job_running))
        for job_id in job_info:
            if all([phase == 'Running' for phase in job_info[job_id]['phase']]):
                x.append(job_info[job_id]['job_type'])
                # r worker的资源请求 维度为 最大作业数量，job请求资源维数, 剩余epoch数量
                if job_info[job_id]['epoch'] and job_info[job_id]['epoch'][0]:
                    r1.append(job_info[job_id]['batch_size'])
                    r2.append(job_info[job_id]['GPUmem_demand'])
                    r3.append(
                        (job_info[job_id]['epoch_num'] - job_info[job_id]['epoch'][0]) / job_info[job_id]['epoch_num'])
                    r4.append(job_info[job_id]['epoch_num'])

                else:
                    r1.append(job_info[job_id]['batch_size'])
                    r2.append(job_info[job_id]['GPUmem_demand'])
                    r3.append(1)
                    r4.append(job_info[job_id]['epoch_num'])
                # w 已分配的worker数量 维度为 最大作业数量
                w.append(job_info[job_id]['worker_num'] - job_info[job_id]['phase'].count('Pending'))
                # d 目前节点上运行worker的情况 维度为 节点数量，最大作业数量
                for node in job_info[job_id]['running_node']:
                    if node:
                        d[node - 1][len(x) - 1] += 1
            else:
                if self.next_job_flag and all([phase == 'Pending' for phase in job_info[job_id]['phase']]):
                    queued_job_score[job_id] = (datetime.datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S')
                                                - datetime.datetime.strptime(job_info[job_id]['creation_time'][0],
                                                                             '%Y-%m-%d %H:%M:%S')).seconds

        # x 转换为onehot
        if not x:
            x.append(0)
        x = np.eye(self.max_job_type)[x]
        x = np.pad(x, ((0, self.max_job_running - len(x)), (0, 0)), 'constant', constant_values=(0, 0))
        # r和w用0填充

        r1 = np.pad(np.array(r1), (0, self.max_job_running - len(r1)), 'constant', constant_values=0)
        r2 = np.pad(np.array(r2), (0, self.max_job_running - len(r2)), 'constant', constant_values=0)
        r3 = np.pad(np.array(r3), (0, self.max_job_running - len(r3)), 'constant', constant_values=0)
        r4 = np.pad(np.array(r4), (0, self.max_job_running - len(r4)), 'constant', constant_values=0)
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
                                            'scheduled_node': np.full(self.max_worker_num, fill_value=-1, dtype=int),
                                            'batch_size': job_info[scheduling_job_id]['batch_size']}
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
                                            'scheduled_node': np.full(self.max_worker_num, fill_value=-0, dtype=int),
                                            'batch_size': 0
                                            }

        jx = np.eye(self.max_job_type)[self.scheduling_job_dict['job_type']]
        js = self.scheduling_job_dict['scheduled_node']
        j.append(self.scheduling_job_dict['batch_size'])
        j.append(self.scheduling_job_dict['GPUmem_demand'])
        j.append(self.scheduling_job_dict['worker_num'])
        j.append(self.scheduling_job_dict['unscheduled_worker_num'])

        state = np.concatenate(
            (np.reshape(node_worker, -1), np.reshape(node_feature, -1), np.reshape(x, -1), np.reshape(r1, -1),
             np.reshape(r2, -1),
             np.reshape(r3, -1), np.reshape(r4, -1),
             np.reshape(w, -1), np.reshape(d, -1),
             np.reshape(np.array(j), -1), np.reshape(jx, -1), js),
        )
        # print(np.shape(state))

        # print(np.shape(np.reshape(x, -1)))
        # print(np.shape(np.reshape(r1, -1)))
        # print(np.shape(np.reshape(r2, -1)))
        # print(np.shape(np.reshape(r3, -1)))
        # print(np.shape(np.reshape(r4, -1)))
        # print(np.shape(np.reshape(w, -1)))
        # print(np.shape(np.reshape(d, -1)))
        # print(np.shape(np.reshape(np.array(j), -1)))
        # print(np.shape(np.reshape(jx, -1)))
        # print(np.shape(np.reshape(js, -1)))

        self.runnin_state.push(state)

        state = (state - self.runnin_state.mean) / (self.runnin_state.std + 0.0000001)
        return state, schedule_flag

    # 提取出等待调度的作业特征
    def make_decision(self, node_info, job_info, node_dict, curr_time, reward, j):
        # 提取出各个节点上所有GPU的使用情况
        GPU_info = dict()
        for node_name in node_info:
            temp = dict()
            temp['GPU_num'] = node_info[node_name]['GPU_num']
            temp['GPU_util'] = np.array(node_info[node_name]['GPU_util'])
            temp['GPUmemory_util'] = np.array(node_info[node_name]['GPUmemory_util'])
            print(temp['GPUmemory_util'])
            temp['GPUmemory_total'] = np.array(node_info[node_name]['GPUmemory_total'])
            temp['GPU_worker'] = list(np.zeros_like(temp['GPU_util']))
            GPU_info[node_name] = temp

        state, schedule_flag = self.state_extraction(node_info, job_info, node_dict, curr_time)
        self.receive_reward(state, reward, j)

        if not schedule_flag:
            action, action_prob = self.select_action(state)
            if action == 0:
                self.temp_state_action.append(
                    {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': False})
                # self.action_writer.add_scalar('step/res', 0, global_step=self.cnt1)
                self.cnt1 += 1
            else:
                self.temp_state_action.append(
                    {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': False})
                # self.action_writer.add_scalar('step/res', -1, global_step=self.cnt1)
                self.cnt1 += 1
                # print('Do Not Need Scheduling')
            return None

        policy = dict()
        if len(self.scheduling_job_dict['pod_list']) != self.scheduling_job_dict['worker_num']:
            return None

        for i in range(len(self.scheduling_job_dict['pod_list'])):
            # print(self.scheduling_job_dict)
            curr_policy = dict()
            state, schedule_flag = self.state_extraction(node_info, job_info, node_dict, curr_time)
            action, action_prob = self.select_action(state)
            if action == 0:
                if i == 0:
                    self.temp_state_action.append(
                        {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': False})
                    # self.action_writer.add_scalar('step/res', 0, global_step=self.cnt1)
                    self.cnt1 += 1
                else:
                    self.temp_state_action.append(
                        {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': False})
                    # not allow gang scheduling
                    # self.action_writer.add_scalar('step/res', -2, global_step=self.cnt1)
                    self.cnt1 += 1
                return None
            # 首先筛选显存要求足够的的GPU，选择一个最空闲的GPU
            # curr_node = GPU_info[self.node_dict[action]]
            #
            # mem_cap = [k for k, x in enumerate(curr_node['GPUmemory_util']) if (
            #             x + self.scheduling_job_dict['GPUmem_demand'] < 0.9 * curr_node['GPUmemory_total'][0])]
            # if len(mem_cap) == 0:
            #     self.temp_state_action.append(
            #         {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': True, 'reward': 0})
            #     # self.action_writer.add_scalar('step/res', -3, global_step=self.cnt1)
            #     self.cnt1 += 1
            #     # print('Exceed Resource Limit')
            #     return None
            # min_worker = 10
            # min_gpu = -1
            # for id in mem_cap:
            #     if curr_node['GPU_worker'][id] < min_worker:
            #         min_gpu = id
            #         min_worker = curr_node['GPU_worker'][id]
            # curr_node['GPU_worker'][min_gpu] += 1
            # self.temp_state_action.append(
            #     {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': False})
            # # self.action_writer.add_scalar('step/res', 0, global_step=self.cnt1)
            # self.cnt1 += 1
            #
            self.scheduling_job_dict['scheduled_node'][self.scheduling_job_dict['worker_num']
                                                       - self.scheduling_job_dict['unscheduled_worker_num']] = action
            self.scheduling_job_dict['unscheduled_worker_num'] -= 1
            # mem_cap = [k for k, x in enumerate(curr_node['GPUmemory_util'])
            #            if (x + self.scheduling_job_dict['GPUmem_demand'] < 0.9 * curr_node['GPUmemory_total'][0])]
            # if GPU_info + self.scheduling_job_dict['GPUmem_demand'] < 0.9 * curr_node['GPUmemory_total'][0]
            curr_policy['namespace'] = self.scheduling_job_dict['namespace']
            curr_policy['node_name'] = self.node_dict[self.gpu_host_array[action][0]]
            curr_policy['job_id'] = self.scheduling_job_dict['job_id']
            curr_policy['node_id'] = self.gpu_host_array[action][0]
            curr_policy['gpu_index'] = self.gpu_host_array[action][1]
            # print(action)
            # print(curr_policy)
            GPU_info[curr_policy['node_name']]['GPU_util'][curr_policy['gpu_index']] += self.scheduling_job_dict[
                'GPUutil_demand']
            GPU_info[curr_policy['node_name']]['GPUmemory_util'][curr_policy['gpu_index']] += self.scheduling_job_dict[
                'GPUmem_demand']

            GPU_info[curr_policy['node_name']]['GPU_worker'][curr_policy['gpu_index']] += 1
            if GPU_info[curr_policy['node_name']]['GPUmemory_util'][curr_policy['gpu_index']] > 0.9 * \
                    GPU_info[curr_policy['node_name']]['GPUmemory_total'][curr_policy['gpu_index']]:
                self.temp_state_action.append(
                    {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': True})
                return None
            self.temp_state_action.append(
                {'state': state, 'action': action, 'action_prob': action_prob, 'step': j, 'exceed_flag': False})

            policy[self.scheduling_job_dict['pod_list'][i]] = curr_policy

        return policy

    def receive_reward(self, next_state, reward, j):

        for sa in self.temp_state_action:
            if j - sa['step'] > self.reward_interval:
                if sa['exceed_flag']:
                    self.store_transition(
                        Transition(sa['state'], sa['action'], sa['action_prob'], -10, next_state, sa['step']))
                else:
                    self.store_transition(
                        Transition(sa['state'], sa['action'], sa['action_prob'], reward, next_state, sa['step']))
                self.action_writer.add_scalar('step/action', sa['action'], global_step=self.cnt2)
                self.action_writer.add_scalar('step/action_prob', sa['action_prob'], global_step=self.cnt2)
                self.cnt2 += 1
        self.update(j)
        self.temp_state_action = list(filter(lambda sa: j - sa['step'] <= self.reward_interval, self.temp_state_action))
        with open('replay_buffer_pkl', 'wb') as f:
            pickle.dump(self.buffer, f)
        return

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.actor_net.eval()
        with torch.no_grad():
            action_prob = self.actor_net(state)
            print(action_prob)
        c = Categorical(action_prob)

        # 这里直接手动覆盖
        c = Categorical(torch.tensor([[0,0.5,0.5]]))
        # random.uniform(0, 1) < 0.5 随机
        if random.uniform(0, 1) < 0.1:
            action = random.choice(range(self.dim_action))
            return action, action_prob[:, action].item()
            # return torch.argmax(action_prob).item(), torch.max(action_prob).item()
        else:
            # return torch.argmax(action_prob).item(), torch.max(action_prob).item()
            action = c.sample()
            return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        self.critic_net.eval()
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net' + str(time.time())[:10] + '.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net' + str(time.time())[:10] + '.pkl')

    def load_param(self, actor_time, critic_time):
        self.actor_net.load_state_dict(torch.load('./param/net_param/actor_net' + actor_time + '.pkl'))
        self.critic_net.load_state_dict(torch.load('./param/net_param/critic_net' + critic_time + '.pkl'))

    def store_transition(self, transition):
        if len(self.buffer) == self.buffer_capacity:
            del self.buffer[:4]
        self.buffer.append(transition)
        self.buffer_counter += 1

    def delete_buffer(self):
        self.buffer = []

    def update(self, i_epoch):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        step = [t.step for t in self.buffer]
        if len(step) == 0:
            return
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        self.actor_net.train()
        self.critic_net.train()
        R = 0
        Gt = []
        last_step = step[len(step) - 1]
        for i in range(len(reward))[::-1]:
            cur_step = step[i]
            if cur_step < last_step:
                R = reward[i] + gamma * R
            Gt.insert(0, R)
            last_step = cur_step
        # for r, s in zip(reward[::-1], step[::-1]):
        #     R = r + gamma * R
        #     Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print(Gt)
        # print(step)
        # print("The agent is updateing....")
        action_loss_list = []
        value_loss_list = []
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, True):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_epoch, self.training_step))
                # if self.training_step % 10000 == 0:
                #     self.save_param()
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                action_prob_list = self.actor_net(state[index])

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                # update actor network
                # action_loss = -torch.min(surr1, surr2).mean() + F.binary_cross_entropy(action_prob_list,
                #                                                                   torch.full_like(action_prob_list, 0.33333333), reduction='mean')  # MAX->MIN desent
                action_loss = -torch.min(surr1, surr2).mean()
                # print(action_loss)
                action_loss_list.append(action_loss.item())

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                value_loss_list.append(value_loss.item())
                # print(value_loss)

                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        self.actor_net.eval()
        self.critic_net.eval()
        if action_loss_list and value_loss_list:
            self.train_writer.add_scalar('train_agent/action_loss', np.mean(np.array(action_loss_list)),
                                         global_step=self.training_cnt)
            self.train_writer.add_scalar('train_agent/value_loss', np.mean(np.array(value_loss_list)),
                                         global_step=self.training_cnt)
        self.training_cnt += 1
        if i_epoch % 10 == 0:
            self.save_param()
        # print('model saved')
        # del self.buffer[:]  # clear experience
