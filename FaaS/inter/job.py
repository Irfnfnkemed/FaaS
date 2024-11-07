import os
from multiprocessing import Pipe, Event
from multiprocessing.connection import Connection
from typing import Callable, Any

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessContext

from .ipc import ClientIPC


class Job:
    def __init__(self):
        self._ipc = ClientIPC()
        self._conn: Connection[Any, Any] = None
        self._process: ProcessContext = None
        self._stop = Event()

    def run(self, func: Callable[..., Any], args: Any):
        self._ipc.connect('localhost', 12345)
        cmd, gpu_list = self._ipc.recv()
        assert cmd == 'alloc-to'
        if len(gpu_list) == 0:
            raise RuntimeError()
        while True:
            self._conn, proc_conn = Pipe()
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
            self._process = mp.spawn(fn=func, args=args + (proc_conn, self._stop,), nprocs=len(gpu_list), join=False)
            while True:
                info = int(self._conn.recv(1024).decode())
                if info == -1:  # training process has ended
                    self._process.join()
                    self._conn.close()
                    return
                elif info > 0:
                    self._ipc.send('alloc-from', info)
                    cmd, new_gpu_list = self._ipc.recv()
                    assert cmd == 'alloc-to'
                    if len(new_gpu_list) == 0:
                        self._conn.send('-1'.encode())  # Request was rejected, accumulate bs on worker
                    else:
                        self._stop.set()
                        self._conn.send('0'.encode())  # Requset was agreed, save checkpoint
                        self._process.join()
                        self._conn.close()
                        self._stop.clear()
                        gpu_list = new_gpu_list
                        break
