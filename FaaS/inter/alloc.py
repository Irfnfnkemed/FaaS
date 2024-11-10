import threading
from typing import List, Dict


class GPU:
    def __init__(self, node_ip: str, device_id: int):
        self.node_ip = node_ip
        self.device_id = device_id
        self.available = True

    def __repr__(self):
        return f"{self.node_ip}:{self.device_id}"


class Node:
    def __init__(self, node_ip: str):
        self.node_ip = node_ip
        self.device_list: Dict[int, GPU] = {}
        self.available = 0

    def set_device(self, device_id: List[int]):
        for gpu_id in device_id:
            if gpu_id not in self.device_list:
                self.device_list[gpu_id] = GPU(self.node_ip, gpu_id)
                self.available += 1


class Allocator:
    def __init__(self):
        self._gpus: Dict[str, Node] = {}  # node_ip -> Node
        self._jobs_occupy: Dict[int, List[GPU]] = {}
        self._lock = threading.Lock()

    def set_device(self, node_ip: str, device_id: List[int]):
        if node_ip not in self._gpus:
            self._gpus[node_ip] = Node(node_ip)
        self._gpus[node_ip].set_device(device_id)

    def alloca(self, job_id: int, num: int) -> List[str]:
        with self._lock:
            if job_id in self._jobs_occupy:
                for gpu in self._jobs_occupy[job_id]:  # Temporary release occupied gpus
                    self._gpus[gpu.node_ip].available += 1
                    gpu.available = True
            sorted_nodes = sorted(self._gpus.values(), key=lambda node: node.available, reverse=True)
            alloca_list = []
            for i in range(1, len(sorted_nodes) + 1):
                max_n = sum(sorted_nodes[j].available for j in range(i))
                if max_n >= num:
                    for node in sorted_nodes:
                        for gpu in node.device_list.values():
                            if gpu.available:
                                gpu.available = False
                                node.available -= 1
                                alloca_list.append(gpu)
                            else:
                                continue
                            if len(alloca_list) == num:
                                self._jobs_occupy[job_id] = alloca_list
                                result = [repr(gpu) for gpu in alloca_list]
                                return result
            if job_id in self._jobs_occupy:
                for gpu in self._jobs_occupy[job_id]:  # Restore the occupied gpus since allocation is failed
                    self._gpus[gpu.node_ip].available -= 1
                    gpu.available = False
            return []

    def free(self, job_id: int):
        with self._lock:
            for gpu in self._jobs_occupy[job_id]:
                self._gpus[gpu.node_ip].available += 1
                gpu.available = True
            self._jobs_occupy.pop(job_id)
