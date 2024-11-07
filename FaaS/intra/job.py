import math
import sys
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .elements import _FaaSStatus, _FaaSOptimizer, _FaaSAdjuster, _FaaSGradMonitor, FaaSDataLoader
from .env import rank, world_size


class IntraOptim:

    def __init__(self, model: torch.nn.Module, trainloader: FaaSDataLoader, testloader: DataLoader,
                 optimizer: torch.optim.Optimizer, accumulation_steps: int, share: Synchronized[int],
                 request_event: Event, response_event: Event):
        self._status = _FaaSStatus(accumulation_steps)
        self._model = DDP(model, device_ids=[rank()])
        self._optimizer = _FaaSOptimizer(optimizer)
        self._grad_monitor = _FaaSGradMonitor(model)
        self._adjuster = _FaaSAdjuster()
        self.trainloader = trainloader
        self.testloader = testloader
        self._share = share
        self._request_event = request_event
        self._response_event = response_event
        self.epochs_now = 0
        self._status.associate(self._model)
        self._optimizer.associate(self._status, self._adjuster, self._grad_monitor)
        self._adjuster.associate(self._status, self._optimizer, self._grad_monitor)
        self._grad_monitor.associate(self._status)
        self.trainloader.associate(self._status)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def is_break(self) -> bool:
        return self._status.is_break

    def sync_or_not(self):
        return self._status.sync_or_not

    def beg_epoch(self):
        if self._status.adapt_bs:
            new_bs = self._adjuster.new_bs  # global new bs
            new_world = math.ceil(new_bs / self._adjuster.bs_upper)
            dist.barrier()
            if new_world != world_size():
                if rank() == 0:
                    self._share = new_world
                    self._request_event.set()
                self._response_event.wait()
                info = self._share
                if rank() == 0:
                    self._response_event.clear()
                assert info == -1 or info == 1
                if info == -1:  # Request was rejected, accumulate bs on worker
                    local_bs = int(new_bs / world_size())
                    accumulation_steps = math.ceil(local_bs / self._adjuster.bs_upper)
                    local_bs = local_bs / accumulation_steps
                    self.trainloader.set_batch_size(local_bs)
                    self._status.accumulation_steps = accumulation_steps
                else:  # Request was agreed, save checkpoint
                    if rank() == 0:
                        pass  # TODO: save checkpoint
                    dist.barrier()
                    self.exit()
            self._status.set_adapt_lr()
            self._adjuster.clear()
            self._grad_monitor.clear()
            self.epochs_now += 1
        elif self._status.adapt_lr:
            self._status.set_adapt_bs()
            self._adjuster.clear()
            self._grad_monitor.clear()

    def exit(self):
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(0)
