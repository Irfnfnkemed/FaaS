import threading
from typing import List, Dict

from .alloc import Allocator
from .ipc import ServerIPC, Server, get_ip


class JobCard:
    def __init__(self, job_id: int, monitor: threading.Thread):
        self.job_id = job_id
        self.monitor = monitor
        self.gpu_num = 0
        self.shortage = 0


class Scheduler:
    def __init__(self, port: int):
        self._allocator = Allocator()
        self._job_cards: Dict[int, JobCard] = {}
        self._cnt = 0
        self._lock = threading.Lock()
        self._server = Server()
        self._server_port = port
        self._adjust_signal = threading.Event()

    def set_device(self, node_ip: str, device_id: List[int]):
        self._allocator.set_device(node_ip, device_id)

    def run(self):
        self._server.serve(get_ip(), self._server_port)
        while True:
            job_ipc = self._server.accept()
            job_id = self.alloc_id()
            gpu_list = self._allocator.alloca(job_id, 1)
            if len(gpu_list) == 0:
                job_ipc.send('alloc', [])
                job_ipc.close()  # TODO: adjust other jobs to accumulate / adjust epb standard
            else:
                job_ipc.send('alloc', gpu_list)
                job_monitor_thread = threading.Thread(target=self.monitor_job, args=(job_id, job_ipc,))
                job_card = JobCard(job_id, job_monitor_thread)
                with self._lock:
                    self._job_cards[job_id] = job_card
                job_monitor_thread.start()

    def monitor_job(self, job_id: int, job_ipc: ServerIPC):
        while True:
            if self._adjust_signal.is_set():
                pass  # TODO:adjust global epb bound

            cmd, data = job_ipc.recv()
            if cmd == 'alloc':
                result = self._allocator.alloca(job_id, int(data))
                with self._lock:
                    if len(result) != 0:
                        self._job_cards[job_id].gpu_num = len(result)
                    else:
                        self._job_cards[job_id].shortage += 1
                job_ipc.send('alloc', result)
            elif cmd == 'end':
                self._allocator.free(job_id)
                job_ipc.close()
                with self._lock:
                    self._job_cards.pop(job_id)
                return

    def alloc_id(self) -> int:
        with self._lock:
            job_id = self._cnt
            self._cnt += 1
            return job_id
