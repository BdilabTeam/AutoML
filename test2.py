from k8s_methods_test1 import podScheduling, podBinding
import kubernetes as k8s
from kubernetes import client
from collections import OrderedDict

k8s.config.load_kube_config(config_file="./kubeconfig.yaml")
k8sCoreV1api = client.CoreV1Api()
if __name__ == "__main__":
    job_tracker = OrderedDict()
    job_tracker['dcgan-7-3-20-16'] = {'gpu_id': [], 'node_id': []}
    job_epoch_records = OrderedDict()
    policy = {'dcgan-7-3-20-16-worker-0': {'namespace': 'dljobs', 'node_name': 'master', 'job_id': 'dcgan-7-3-20-16', 'node_id': 1, 'gpu_index': 0},
              'dcgan-7-3-20-16-worker-1': {'namespace': 'dljobs', 'node_name': 'master', 'job_id': 'dcgan-7-3-20-16', 'node_id': 1, 'gpu_index': 0},
              'dcgan-7-3-20-16-worker-2': {'namespace': 'dljobs', 'node_name': 'master', 'job_id': 'dcgan-7-3-20-16', 'node_id': 1, 'gpu_index': 0}}

    podScheduling(k8sCoreV1api, policy, job_tracker, job_epoch_records)
