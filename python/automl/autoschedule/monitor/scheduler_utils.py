import heapq
from enum import Enum, unique

from kubernetes.client import ApiException

@unique
class PodPhase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"

def get_ready_node(k8sCoreV1api):
    node_list = k8sCoreV1api.list_node()  # 列出集群中所有节点的信息，包括节点的名称，ip，资源利用率等
    ready_node = []
    for node in node_list.items:  # for i in list[V1Node]
        # 可用的节点：the node is healthy and ready to accept pods
        if node.status.conditions[-1].status == "True" and node.status.conditions[-1].type == "Ready":
            ready_node.append(node.metadata.name)  # V1Node.metadata.name

    return ready_node


def get_pod_by_namespace_and_phase(k8sCoreV1api,namespace, pod_phase: PodPhase):
    podInstance = k8sCoreV1api.list_namespaced_pod(namespace)
    scheduledPodName = []
    try:
        for i in podInstance.items:  # for i in list[V1Pod]
            # pod期望状态:spec，pod当前实际状态：status
            if i.status.phase == pod_phase.value:
                scheduledPodName.append(i)
    except ApiException as e:
        print("Exception when calling CoreV1Api->list_pod_for_all_namespaces: %s\n" % e)
    return scheduledPodName

def get_pod_by_task(k8sCoreV1api,task_name,namespace="test"):
    task_pods = []
    try:
        # 获取指定命名空间中的所有 Pods
        pods = k8sCoreV1api.list_namespaced_pod(namespace)
        for pod in pods.items:
            # 检查 Pod 名称和状态
            if pod.metadata.name.startswith(task_name) and pod.status.phase == "Pending":
                task_pods.append(pod.metadata.name)
    except ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")

    return task_pods


def get_gpu_index(host_info,threshold):
    gpu_result_info = []

    for ip, info in host_info.items():
        gpu_memory_free = info.get("gpu_memory_free", None)

        if gpu_memory_free:
            for gpu_index, free_memory in enumerate(gpu_memory_free):
                if free_memory > threshold:
                    heapq.heappush(gpu_result_info, (free_memory - threshold, ip + "-" + str(gpu_index)))

    if gpu_result_info:
        host_ip, gpu_index = heapq.heappop(gpu_result_info)[1].split("-")
        print(host_ip, gpu_index)
        return host_ip, gpu_index

    return "", -1
