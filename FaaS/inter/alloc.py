import socket
import threading
from typing import List, Dict

from .ipc import ServerIPC, Server


class Alloc:
    class Job_card:
        def __init__(self, id: int, monitor: threading.Thread):
            self._id = id
            self._monitor = monitor
            self._gpu_list = []
            self._shortage = 0

    def __init__(self, gpu_id: List[int]):
        self._job_cards: Dict[int, Alloc.Job_card] = {}
        self._cnt = 0
        self._lock = threading.Lock()
        self._server = Server()
        self._adjust_signal = threading.Event()
        self._gpu_available = {id: True for id in gpu_id}

    def run(self):
        self._server.serve()
        while True:
            job_ipc = self._server.accept()
            gpu_id = self.request_one()
            if gpu_id == -1:
                job_ipc.send('alloc-to', [])
                job_ipc.close()  # TODO: adjust other to accumulate
                continue
            else:
                job_ipc.send('alloc-to', [gpu_id])
            with self._lock:
                id = self._cnt
                self._cnt += 1
                job_monitor_thread = threading.Thread(target=self.monitor_job, args=(id, job_ipc,))
                self._job_cards[id] = Alloc.Job_card(id, job_monitor_thread)
            job_monitor_thread.start()

    def monitor_job(self, id: int, job_ipc: ServerIPC):
        while True:
            if self._adjust_signal.is_set():
                pass  # TODO:adjust global epb bound

            cmd, data = job_ipc.recv()
            if cmd == 'alloc-from':
                result = self.request_resource(id, int(data))
                job_ipc.send('alloc-to', result)
            elif cmd == 'end-from':
                self.release_resource(id)
                job_ipc.close()

    def request_one(self) -> int:
        with self._lock:
            for gpu_id in list(self._gpu_available.keys()):
                if self._gpu_available[gpu_id]:
                    return gpu_id
        return -1

    def request_resource(self, id: int, num: int) -> List[int]:
        with self._lock:
            if len(self._job_cards[id]._gpu_list) > num:
                new_gpu_list = self._job_cards[id]._gpu_list[:num]
                for gpu_id in list(self._gpu_available.keys()):
                    if gpu_id not in new_gpu_list:
                        self._gpu_available[gpu_id] = True
                self._job_cards[id]._gpu_list = new_gpu_list
                self._job_cards[id]._shortage = 0
                return self._job_cards[id]._gpu_list
            else:
                cnt = num - len(self._job_cards[id]._gpu_list)
                select = []
                for gpu_id in list(self._gpu_available.keys()):
                    if self._gpu_available[gpu_id]:
                        select.append(gpu_id)
                        cnt -= 1
                    if cnt == 0:
                        break
                if cnt > 0:
                    self._job_cards[id]._shortage += 1
                    return []
                else:
                    for gpu_id in select:
                        self._job_cards[id]._gpu_list.append(gpu_id)
                        self._gpu_available[gpu_id] = False
                    return self._job_cards[id]._gpu_list

    def release_resource(self, id: int):
        with self._lock:
            for gpu_id in self._job_cards[id]._gpu_list:
                self._gpu_available[gpu_id] = True
            self._job_cards.pop(id)
