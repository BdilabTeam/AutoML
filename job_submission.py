import os.path

from fabric import Connection
from fabric.runners import Result
from fabric import SerialGroup as Group
from fabric.group import GroupResult
import re
import json
from datetime import datetime
import numpy.random as random
import pickle
from collections import OrderedDict

import my_logging

if os.path.exists("job_submit.log"):
    os.remove("job_submit.log")

mylog = my_logging.MyLogger(filename="job_submit.log",level="debug")
# from fabric import ThreadingGroup as Group

hostinfo = dict()
hostinfo[0] = {'ip': '192.168.1.77', 'user': 'root', 'password': '123456', 'dir': '/27T/lwj/dljobs/jobs'}
hostinfo[1] = {'ip': '192.168.1.66', 'user': 'root', 'password': '123456', 'dir': '/home/project/lwj/dljobs/jobs'}
# hostinfo[2] = {'ip': '192.168.1.88', 'user': 'root', 'password': '123456', 'dir': '/home/project/lwj/dljobs/jobs'}
hostinfo[2] = {'ip': '192.168.1.99', 'user': 'root', 'password': '123456', 'dir': '/home/project/lwj/dljobs/jobs'}


# job_name_list = {'resnet': 'ResNet50', 'vgg': 'VGG16', 'vgg': 'CNN', 'unet': 'UNet'}


class jobSubmission():
    def connect(self):
        self.connections = list()
        for host in self.hostinfo:
            conn = Connection(self.hostinfo[host]['ip'], user=self.hostinfo[host]['user'], port=22,
                              connect_kwargs={"password": self.hostinfo[host]['password']})
            conn.run('hostname')  # 这行不能忽略
            self.connections.append(conn)
        # for conn in self.connections:
        #     res = conn.run('ntpdate cn.pool.ntp.org', hide=True)
        return 0

    def submit_job(self, node_index, job_name):
        path = self.hostinfo[node_index]['dir']
        kubeconfig_path = "/etc/kubernetes/admin.conf"

        with self.connections[node_index].cd(path):
            with self.connections[node_index].prefix(f'export KUBECONFIG={kubeconfig_path}'):
                mylog.logger.info(f'command is `kubectl create -f `{job_name}.yaml`')
                res = self.connections[node_index].run('kubectl create -f {}.yaml'.format(job_name), hide=True)
        return res

    def delete_job(self, node_index, job_name):
        path = self.hostinfo[node_index]['dir']
        with self.connections[node_index].cd(path):
            res = self.connections[node_index].run('kubectl delete -f {}.yaml'.format(job_name), hide=True)
        return res

    def submit_all(self, node_index):
        for job_name in self.job_name_list:
            self.submit_job(node_index, job_name)

    def submit_list(self, node_index, name_list):
        for job_name in name_list:
            self.submit_job(node_index, job_name)

    def delete_all(self, node_index):
        for job_name in self.job_name_list:
            self.delete_job(node_index, job_name)

    def __init__(self, hostinfo, job_name_list):
        self.hostinfo = hostinfo
        self.job_name_list = job_name_list
        self.connect()


if __name__ == '__main__':
    # job_name_list = ['vgg-1-3-30-16', 'vgg-1-2-50-32', 'resnet-2-2-30-8', 'resnet-2-3-50-16', 'unet-3-2-30-8',
    #                  'unet-3-3-50-16', 'cnn-4-4-10-16', 'cnn-4-5-20-32', 'rnn-5-2-30-8', 'rnn-5-3-50-16',
    #                  'cnnl-6-2-30-16', 'cnnl-6-3-50-32']
    # job_name_list = ['vgg-1-3-30-16', 'vgg-1-2-50-32', 'vgg-1-4-40-16', 'resnet-2-2-30-8', 'resnet-2-3-50-16',
    #                  'resnet-2-4-40-8', 'unet-3-2-30-8', 'unet-3-3-50-16', 'unet-3-4-40-8', 'cnn-4-4-10-16',
    #                  'cnn-4-5-20-32', 'cnn-4-3-10-32']
    job_name_list = []
    with open('cur_submission_dict.pkl', 'rb') as f:
        submission_dict = pickle.load(f)

    submission_dict = OrderedDict(submission_dict)

    for key in submission_dict:
        print(submission_dict[key])
        for job in submission_dict[key]:
            job_name_list.append(job)

    print(job_name_list)
    # sample_poisson = random.poisson(60, size=len(job_name_list))
    # # job_name_list = {'resnet', }
    submission = jobSubmission(hostinfo, job_name_list)
    # submission.submit_all(0)
    submission.delete_all(0)
