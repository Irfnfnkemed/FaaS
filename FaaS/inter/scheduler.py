import threading
import time
from typing import List, Dict

from .alloc import Allocator
from .ipc import ServerIPC, Server, get_ip


class JobCard:
    def __init__(self, job_id: int, monitor: threading.Thread):
        self.job_id = job_id
        self.monitor = monitor
        self.gpu_num = 0
        self.scarcity = 0
        self.surplus = 0
        self.adjust_standard = False


class Scheduler:
    def __init__(self, port: int):
        self._allocator = Allocator()
        self._job_cards: Dict[int, JobCard] = {}
        self._cnt = 0
        self._lock = threading.Lock()
        self._server = Server()
        self._server_port = port
        
        self._epb_standard = 1.0
        self._epb_standard_upper = 2.0
        self._epb_standard_lower = 0.5
        self._epb_standard_adjust_rate = 0.95

    def set_device(self, node_ip: str, device_id: List[int]):
        self._allocator.set_device(node_ip, device_id)

    def run(self):
        self._server.serve(get_ip(), self._server_port)
        adjust_standard_thread = threading.Thread(target=self.adjust_standard, args=())
        adjust_standard_thread.start()
        while True:

            job_ipc = self._server.accept()
            job_id = self.alloc_id()
            cmd, _ = job_ipc.recv()
            assert cmd == 'shakehands'
            job_ipc.send('shakehands', job_id)
            cmd, data = job_ipc.recv()
            assert cmd == 'alloc'
            gpu_list = self._allocator.alloca(job_id, int(data))
            if len(gpu_list) == 0:
                # TODO: adjust other jobs to accumulate / adjust epb standard
                job_ipc.close()  
            else:
                job_ipc.send('alloc', gpu_list)
                job_monitor_thread = threading.Thread(target=self.monitor_job, args=(job_id, job_ipc,))
                job_card = JobCard(job_id, job_monitor_thread)
                with self._lock:
                    self._job_cards[job_id] = job_card
                job_monitor_thread.start()

    def monitor_job(self, job_id: int, job_ipc: ServerIPC):
        while True:
            cmd, data = job_ipc.recv()
            if cmd == 'alloc':
                result = self._allocator.alloca(job_id, int(data))
                with self._lock:
                    if len(result) < int(data): # Lack of resources
                        self._job_cards[job_id].scarcity += 1
                        self._job_cards[job_id].surplus = 0
                    else:
                        self._job_cards[job_id].scarcity = 0
                        self._job_cards[job_id].surplus = 0
                    self._job_cards[job_id].gpu_num = len(result)    
                job_ipc.send('alloc', result)
            elif cmd == 'free':
                result = self._allocator.alloca(job_id, int(data))
                assert len(result) == int(data)
                with self._lock:
                    self._job_cards[job_id].scarcity = 0
                    self._job_cards[job_id].surplus += 1
                    self._job_cards[job_id].gpu_num = int(data)
                job_ipc.send('free', result)
            elif cmd == 'end':
                self._allocator.free(job_id)
                job_ipc.close()
                with self._lock:
                    self._job_cards.pop(job_id)
                return
            elif cmd == 'standard':
                standard = 1.0
                with self._lock:
                    standard = self._epb_standard
                    self._job_cards[job_id].adjust_standard = True
                job_ipc.send('standard', standard)
            elif cmd == 'ideal':
                with self._lock:
                    self._job_cards[job_id].scarcity = 0
                    self._job_cards[job_id].surplus = 0

    def alloc_id(self) -> int:
        with self._lock:
            job_id = self._cnt
            self._cnt += 1
            return job_id
        
    def adjust_standard(self):
        while True:
            with self._lock:
                scarcity = 0
                surplus = 0
                already_adjust = 0
                for job_card in self._job_cards.values():
                    if job_card.scarcity >= 1:
                        scarcity += 1
                    if job_card.surplus >= 1:
                        surplus += 1
                    if job_card.adjust_standard:
                        already_adjust += 1
                if already_adjust > len(self._job_cards) / 2: # Adjustment of epb-standard has been responded
                    if scarcity > len(self._job_cards) / 2: 
                        for job_card in self._job_cards.values():
                            job_card.adjust_standard = False
                        self._epb_standard = min(self._epb_standard_upper, self._epb_standard / self._epb_standard_adjust_rate) 
                    elif surplus > len(self._job_cards) / 2: 
                        for job_card in self._job_cards.values():
                            job_card.adjust_standard = False
                        self._epb_standard = max(self._epb_standard_lower, self._epb_standard * self._epb_standard_adjust_rate)    
            time.sleep(10)
                
