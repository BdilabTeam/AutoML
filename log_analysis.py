import json
from collections import OrderedDict
import numpy as np
import datetime
import pandas as pd
import os
import job_name_constants

# L 为模型类别数量
L = 6
# K为考虑资源的总数，这里我们选择GPU利用率和GPU存储
K = 2
# M为节点总数
M = 2

node_list = ['master', 'node1', 'node2', 'node3']
# 将日志的值中所有时间调整为秒数
def log_time_tuning(log, start_time):
    for tm in log:
        temp = log[tm]['job_info']
        for job_id in temp:
            tm_log = temp[job_id]
            tmp = []
            for time in tm_log['start_time']:
                if time:
                    if datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') > start_time:
                        tmp.append(int((datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') - start_time).seconds))
                    else:
                        tmp.append(-1 * int((start_time - datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')).seconds))
                else:
                    tmp.append(None)
            log[tm]['job_info'][job_id]['start_time'] = tmp

            tmp = []
            for time in tm_log['creation_time']:
                if time:
                    if datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') > start_time:
                        tmp.append(int((datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') - start_time).seconds))
                    else:
                        tmp.append(-1 * int((start_time - datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')).seconds))
                else:
                    tmp.append(None)
            log[tm]['job_info'][job_id]['creation_time'] = tmp

            tmp = []
            for time in tm_log['epoch_time']:
                if time:
                    if datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') > start_time:
                        tmp.append(int((datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') - start_time).seconds))
                    else:
                        tmp.append(-1 * int((start_time - datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')).seconds))
                else:
                    tmp.append(None)
            log[tm]['job_info'][job_id]['epoch_time'] = tmp

            tmp = []
            for time in tm_log['end_time']:
                if time:
                    if datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') > start_time:
                        tmp.append(int((datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') - start_time).seconds))
                    else:
                        tmp.append(int((start_time - datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')).seconds))
                else:
                    tmp.append(None)
            log[tm]['job_info'][job_id]['end_time'] = tmp
    return log

# 提取节点信息
def node_tracking(log):
    util_list = []
    node_log = OrderedDict()
    for tm in log:
        if log[tm]['node_info']:
            tmp = log[tm]['node_info']
            # for node in tmp:
            #     del tmp[node]['CPU_core']
            node_log[tm] =tmp

    return node_log

# 提取指定job的信息
def job_tracking(log, job_id):
    job_log = OrderedDict()

    for tm in log:

        if not log[tm]['job_info'].__contains__(job_id):
            continue

        tm_log = log[tm]['job_info'][job_id]
        if any([phase == 'Running' for phase in tm_log['phase']]):
            # del tm_log['CPU_usage']
            # del tm_log['memory_usage']
            del tm_log['job_index']
            del tm_log['namespace']
            # del tm_log['CPU_demand']
            # del tm_log['memory_demand']
            del tm_log['GPUutil_demand']
            del tm_log['GPUmem_demand']
            job_log[tm] = tm_log
    return job_log

# 提取所有调度动作的运行时信息
def policy_tracking(log):
    policy_log = OrderedDict()
    job_allocation = OrderedDict()
    for tm in log:
        if log[tm]['policy']:
            tmp = log[tm]['policy']
            for pod in tmp:
                if tmp[pod]['job_id'] not in job_allocation:
                    job_allocation[tmp[pod]['job_id']] = OrderedDict()
                    job_allocation[tmp[pod]['job_id']]['node_name'] = []
                    job_allocation[tmp[pod]['job_id']]['gpu_index'] = []
                job_allocation[tmp[pod]['job_id']]['node_name'].append(tmp[pod]['node_name'])
                job_allocation[tmp[pod]['job_id']]['gpu_index'].append(tmp[pod]['gpu_index'])
            policy_log[tm] = tmp
    return policy_log, job_allocation


def job_end_tracking(log, job_id):
    job_makespan_log = []
    job_completion_log = []
    for tm in log:
        if not log[tm]['job_info'].__contains__(job_id):
            continue
        tm_log = log[tm]['job_info'][job_id]
        if any([phase == 'Succeeded' for phase in tm_log['phase']] or [phase == 'Failed' for phase in tm_log['phase']]):
            for i in tm_log['end_time']:
                if i:
                    job_makespan_log.append(i)
                    # if tm_log['start_time']:
                    #     job_completion_log.append(i-tm_log['start_time'][0])
    if job_makespan_log == []:
        return 0
    return max(job_makespan_log)


if __name__ == '__main__':
    log_dir_prefix = 'log/'
    experiment_name = 'RL-test'
    # time_stamps = ['20211113230331', '20211114035111']
    # job_list = ['resnet']
    # job_list = ['resnet']
    # job_list = ['rnn']
    # job_list = ['cnn']
    job_list = job_name_constants.job_name_list
    # job_list = ['vgg']
    log_paths = os.listdir(log_dir_prefix)
    jct=[]
    makespan = []
    for id in log_paths:
        # print(id)
        log_dir = os.path.join(log_dir_prefix, id[:-5])

        log_path = os.path.join(log_dir_prefix, id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 读入log文件
        print(log_path)
        log = json.load(open(log_path), object_pairs_hook=OrderedDict)
        # 将log文件中的所有键更改为从0开始的秒数
        start_time = datetime.datetime.strptime(list(log.keys())[0], '%Y-%m-%d %H:%M:%S')
        reg_log = OrderedDict()
        for tm in log:
            td = (datetime.datetime.strptime(tm, '%Y-%m-%d %H:%M:%S')-start_time).seconds
            reg_log[td] = log[tm]
        # 将log文件中的所有值更改为从0开始的秒数
        reg_log = log_time_tuning(reg_log, start_time)
        json_str = json.dumps(reg_log, indent=4)
        with open(os.path.join(log_dir, 'log.json'), 'w') as path:
            path.write(json_str)

        log = json.load(open(os.path.join(log_dir, 'log.json')), object_pairs_hook=OrderedDict)

        # 提取所有节点的信息
        node_log = node_tracking(log)
        json_str = json.dumps(node_log, indent=4)
        with open(os.path.join(log_dir, 'node.json'), 'w') as path:
             path.write(json_str)

        # 提取所有调度动作的运行时信息
        policy_log, job_allocation = policy_tracking(log)
        print(policy_log)
        json_str = json.dumps(policy_log, indent=4)
        with open(os.path.join(log_dir, 'policy.json'), 'w') as f:
            f.write(json_str)
        json_str = json.dumps(job_allocation, indent=4)
        with open(os.path.join(log_dir, 'job_allocation.json'), 'w') as f:
            f.write(json_str)
        scheduled_log = OrderedDict()
        for tm in policy_log:
            tmp_jobs = policy_log[tm]
            for job_name in tmp_jobs:
                scheduled_log[tmp_jobs[job_name]['job_id']] = int(tm)
                break
        print(scheduled_log)

        # 提取指定job的运行时信息,并将所有的时间信息转化成秒数
        for job_id in job_list:
            job_log = job_tracking(log, job_id)
            json_str = json.dumps(job_log, indent=4)
            with open(os.path.join(log_dir, '{}.json'.format(job_id)), 'w') as f:
                f.write(json_str)
        acc_makespan = 0
        end_log = OrderedDict()
        end_log['max'] = 0
        for job_id in job_list:
            end_log[job_id] = job_end_tracking(log, job_id)
            acc_makespan += end_log[job_id]
            if end_log[job_id] > end_log['max']:
                end_log['max'] = end_log[job_id]

        end_log['total'] = acc_makespan
        json_str = json.dumps(end_log, indent=4)
        with open(os.path.join(log_dir, 'end_time.json'), 'w') as f:
           f.write(json_str)

        # # 分析job的运行信息, 适用于standalone实验
        # csv_path = os.path.join(log_dir, id[:-5] + '.csv')
        # open(csv_path, 'w').close()  # 清空
        # header = ['job_id', 'worker_index', 'job_meta', 'worker_num', 'creation_time', 'start_time', 'running_node', 'gpu_id',
        #           'Avg. epoch interval', 'Std. epoch interval', 'Avg. gpu memory usage', 'Avg. gpu util usage', 'end_time',
        #           'makespan', 'waiting_time']
        # df = pd.DataFrame(header).T
        # df.to_csv(csv_path, index=False, header=False)
        #
        # for job_id in job_list:
        #     data = []
        #     log_path = os.path.join(log_dir, '{}.json'.format(job_id))
        #     log = json.load(open(log_path), object_pairs_hook=OrderedDict)
        #     job_meta = job_id.split('-')
        #     worker_num = int(job_meta[2])
        #     for worker_index i n range(worker_num):
        #         row = {'job_id': job_id, 'worker_index': worker_index, 'job_meta': job_meta, 'worker_num': worker_num,}
        #         epoch_stamp = []
        #         epoch_num = []
        #         mem_usage = []
        #         util_usage = []
        #         for tm in log:
        #             if worker_index in log[tm]['epoch']:
        #             if log[tm]['epoch'][worker_index] and log[tm]['epoch'][worker_index] > len(epoch_stamp):
        #                 if log[tm]['epoch_time'][worker_index]:
        #                     epoch_stamp.append(log[tm]['epoch_time'][worker_index])
        #                     epoch_num.append(log[tm]['epoch'][worker_index])
        #                 if log[tm]['gpumemory_used'][worker_index]:
        #                     mem_usage.append(log[tm]['gpumemory_used'][worker_index])
        #                 if log[tm]['corr_gpu_util'][worker_index]:
        #                     util_usage.append(log[tm]['corr_gpu_util'][worker_index])
        #             row['creation_time'] = log[tm]['creation_time'][0]
        #             row['start_time'] = log[tm]['start_time'][worker_index]
        #             row['running_node'] = log[tm]['running_node'][worker_index]
        #             row['gpu_id'] = log[tm]['gpu_id'][worker_index]
        #         epoch_intervals = []
        #         for i in range(len(epoch_stamp) - 1):
        #             if epoch_stamp[i + 1] > epoch_stamp[i]:
        #                 epoch_intervals.append((epoch_stamp[i + 1] - epoch_stamp[i])/(epoch_num[i+1] - epoch_num[i]))
        #         if len(epoch_intervals) > 0:
        #             # print(epoch_intervals)
        #             row['Avg. epoch interval'] = np.mean(np.array(epoch_intervals))
        #             row['Std. epoch interval'] = np.std(np.array(epoch_intervals))
        #         else:
        #             row['Avg. epoch interval'] = -1
        #             row['Std. epoch interval'] = -1
        #         row['Avg. gpu memory usage'] = np.mean(np.array(mem_usage))
        #         row['Avg. gpu util usage'] = np.mean(np.array(util_usage))
        #         row['end_time'] = end_log[job_id]
        #         row['makespan'] = row['end_time'] - row['creation_time']
        #         row['waiting_time'] = row['start_time'] - row['creation_time']
        #         data.append(row)
        #     df = pd.DataFrame(data)
        #     df.to_csv(csv_path, index=False, header=False, mode='a+')
        print(end_log['total'])
        print(end_log['max'])
        jct.append(end_log['total'])
        makespan.append(end_log['max'])

    print(np.mean(np.array(jct)))
    print(len(jct))
    print(np.mean(np.array(makespan)))
    print(len(makespan))







