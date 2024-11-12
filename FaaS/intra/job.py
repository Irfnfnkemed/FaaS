import math
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .elements import _FaaSStatus, _FaaSOptimizer, _FaaSAdjuster, _FaaSGradMonitor, FaaSDataLoader
from .env import rank, world_size, local_rank
from ..inter.ipc import ClientIPC


class IntraOptim:

    def __init__(self, model: torch.nn.Module, trainloader: FaaSDataLoader, testloader: DataLoader,
                 optimizer: torch.optim.Optimizer, epochs: int, accumulation_steps: int, proxy_ip: str, proxy_port: int):
        if not dist.is_available() or not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=world_size(), rank=rank())
        self._status = _FaaSStatus(accumulation_steps)
        self._model = DDP(model, device_ids=[rank()])
        self._optimizer = _FaaSOptimizer(optimizer)
        self._grad_monitor = _FaaSGradMonitor()
        self._adjuster = _FaaSAdjuster()
        self.trainloader = trainloader
        self.testloader = testloader
        self._ipc = ClientIPC()
        self._epochs = epochs
        self._epochs_now = 0
        self._local_bs = 0
        self._status.associate(self._model)
        self._optimizer.associate(self._status, self._adjuster, self._grad_monitor)
        self._adjuster.associate(self._status, self._optimizer, self._grad_monitor)
        self._grad_monitor.associate(self._model, self._status)
        self.trainloader.associate(self._status)
        if rank() == 0:
            self._ipc.connect(proxy_ip, proxy_port)

    def save_checkpoint(self):
        checkpoint = {
            'model': self._model.module.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'status': self._status.save_state(),
            'epochs_now': self._epochs_now,
            'bs': self._local_bs
        }
        torch.save(checkpoint, './checkpoint.pth')

    def load_checkpoint(self):
        if os.path.exists('./checkpoint.pth'):
            checkpoint = torch.load('./checkpoint.pth')
            self._epochs_now = checkpoint['epochs_now']
            self._model.module.load_state_dict(checkpoint['model'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._status.load_state(checkpoint['status'])
            self.trainloader.set_batch_size(checkpoint['bs'])
            self._local_bs = checkpoint['bs']

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
        if self._epochs_now >= self._epochs:
            self.exit()
            return
        # print(f"LR{rank()},{self._optimizer.param_groups[0]['lr']}")
        if self._status.adapt_bs:
            new_bs = self._adjuster.adjust_bs(self._grad_monitor.epb,
                                              self._local_bs * self._status.accumulation_steps * world_size())  # global new bs
            new_world_size = math.ceil(new_bs / self._adjuster.bs_upper)
            if new_world_size != world_size():
                alloc_result = torch.tensor(0).to(local_rank())
                if rank() == 0:
                    self._ipc.send("alloc", new_world_size)
                    cmd, data = self._ipc.recv()
                    assert cmd == 'alloc'
                    alloc_result = torch.tensor(int(data)).to(local_rank())
                dist.broadcast(alloc_result, src=0)
                if int(alloc_result.item()) == 0:  # Request was rejected, accumulate bs on worker
                    accumulation_steps = math.ceil(new_bs / world_size() / self._adjuster.bs_upper)
                    self._local_bs = int(new_bs / world_size() / accumulation_steps)
                    self._status.accumulation_steps = accumulation_steps
                else:  # Request was agreed, save checkpoint
                    if rank() == 0:
                        self._local_bs = int(new_bs / new_world_size)
                        self._status.accumulation_steps = 1
                        self._status.set_adapt_lr()
                        self.save_checkpoint()
                        pass  # save checkpoint
                    dist.barrier()  # Ensure exiting after checkpoint was saved
                    self.exit()
                    return
            else:  # Directly adjust bs and accumulation-steps
                accumulation_steps = math.ceil(new_bs / world_size() / self._adjuster.bs_upper)
                self._local_bs = int(new_bs / world_size() / accumulation_steps)
                self.trainloader.set_batch_size(self._local_bs)
                self._status.accumulation_steps = accumulation_steps
                # print(f"Adjust{rank()},{local_bs},{accumulation_steps}")
            self._status.set_adapt_lr()
            self._adjuster.clear()
            self._grad_monitor.clear()
        elif self._status.adapt_lr:
            self._epochs_now += 1
            self._status.set_adapt_bs()
            self._adjuster.clear()
            # print(f"EPT{rank()},{self._grad_monitor.ept}")
            self._grad_monitor.clear()

    def get_epoch(self) -> int:
        return self._epochs_now

    def exit(self):
        # print("exit", rank())
        if rank() == 0:
            self._ipc.send('end', '')
            self._ipc.close()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(0)
