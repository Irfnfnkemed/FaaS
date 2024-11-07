import os
from multiprocessing import Event, Value
from typing import Callable, Any

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessContext

from .ipc import ClientIPC


class Job:
    def __init__(self):
        self._ipc = ClientIPC()
        self._request_event = Event()
        self._response_event = Event()
        self._share = Value('i', 0)
        self._process: ProcessContext = None

    def run(self, func: Callable[..., Any], args: Any):
        self._ipc.connect('localhost', 12345)
        cmd, gpu_list = self._ipc.recv()
        assert cmd == 'alloc-to'
        if len(gpu_list) == 0:
            raise RuntimeError()
        while True:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
            self._process = mp.spawn(fn=func, args=args + (self._share, self._request_event, self._response_event,),
                                     nprocs=len(gpu_list), join=False)
            while True:
                self._request_event.wait()
                info = self._share
                self._request_event.clear()
                if info == -1:  # training process has ended
                    self._process.join()
                    return
                elif info > 0:
                    self._ipc.send('alloc-from', info)
                    cmd, new_gpu_list = self._ipc.recv()
                    assert cmd == 'alloc-to'
                    if len(new_gpu_list) == 0:
                        self._share = -1  # Request was rejected, accumulate bs on worker
                        self._response_event.set()
                    else:
                        self._share = 1  # Request was agreed, save checkpoint
                        self._response_event.set()
                        self._process.join()
                        gpu_list = new_gpu_list
                        break
