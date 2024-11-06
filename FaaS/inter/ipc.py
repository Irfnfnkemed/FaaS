import json
import socket
import threading
from typing import List, Dict


class IPC:
    class Job_card:
        def __init__(self, id: int, conn: socket.socket, ip: socket._RetAddress, monitor: threading.Thread):
            self._id = id
            self._conn = conn
            self._ip = ip
            self._monitor = monitor
            self._gpu_list = []
            self._shortage = 0

    def __init__(self, gpu_id: List[int]):
        self._job_cards: Dict[int, IPC.Job_card] = {}
        self._cnt = 0
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._adjust_signal = threading.Event()
        self._gpu_available = {id: True for id in gpu_id}

    def run(self):
        self._server.bind(('localhost', 12345))
        self._server.listen(5)
        while True:
            job_conn, job_ip = self._server.accept()
            gpu_id = self.request_one()
            if gpu_id == -1:
                job_conn.send(json.dumps({'cmd': 'alloc', 'data': []}).encode())
                job_conn.close()  # TODO: adjust other to accumulate
                continue
            else:
                job_conn.send(json.dumps({'cmd': 'alloc', 'data': [gpu_id]}).encode())
            with self._lock:
                id = self._cnt
                self._cnt += 1
                job_monitor_thread = threading.Thread(target=self.monitor_job, args=(id, job_conn,))
                self._job_cards[id] = IPC.Job_card(id, job_conn, job_ip, job_monitor_thread)
            job_monitor_thread.start()

    def monitor_job(self, id: int, job_conn: socket.socket):
        while True:
            if self._adjust_signal.is_set():
                pass  # TODO:adjust global epb bound

            request = job_conn.recv(1024)
            received_data = json.loads(request.decode())
            if received_data['cmd'] == 'request':
                result = self.request_resource(id, int(received_data['data']))
                job_conn.send(json.dumps({'cmd': 'alloc', 'data': result}).encode())
            elif received_data['cmd'] == 'end':
                self.release_resource(id)
                job_conn.close()

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
