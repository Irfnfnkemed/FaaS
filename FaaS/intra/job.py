import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .elements import _FaaSStatus, _FaaSOptimizer, _FaaSAdjuster, _FaaSGradMonitor, FaaSDataLoader
from .env import rank, world_size, local_rank
from ..inter.ipc import ClientIPC


class IntraOptim:

    def __init__(self, model: torch.nn.Module, trainloader: FaaSDataLoader, optimizer: torch.optim.Optimizer, 
                 epochs: int, accumulation_steps: int, proxy_ip: str, proxy_port: int):
        if not dist.is_available() or not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=world_size(), rank=rank())
        self._status = _FaaSStatus(math.ceil(epochs / 5), accumulation_steps)
        self._model = DDP(model, device_ids=[rank()])
        self._optimizer = _FaaSOptimizer(optimizer)
        self._grad_monitor = _FaaSGradMonitor()
        self._adjuster = _FaaSAdjuster()
        self.trainloader = trainloader
        self._ipc = ClientIPC()
        self._epochs = epochs
        self._epochs_now = 0
        self._global_bs = self.trainloader.batch_size * world_size()
        self._status.associate(self._model)
        self._optimizer.associate(self._status, self._adjuster, self._grad_monitor)
        self._adjuster.associate(self._status, self._optimizer, self._grad_monitor)
        self._grad_monitor.associate(self._model, self._status)
        self.trainloader.associate(self._status)
        self.job_id = torch.tensor(0).to(local_rank())
        if rank() == 0:
            self._ipc.connect(proxy_ip, proxy_port)
            self._ipc.send('shakehands', '')
            cmd, job_id = self._ipc.recv()
            assert cmd == 'shakehands'
            self.job_id = torch.tensor(int(job_id)).to(local_rank())
        dist.broadcast(self.job_id, src=0)
        self.load_checkpoint()
        self._beg = True

    def save_checkpoint(self):
        checkpoint = {
            'model': self._model.module.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'status': self._status.save_state(),
            'epochs_now': self._epochs_now,
            'global_bs': self._global_bs
        }
        torch.save(checkpoint, f'./tmp_{self.job_id.item()}/checkpoint.pth')

    def load_checkpoint(self):
        if os.path.exists(f'./tmp_{self.job_id.item()}/checkpoint.pth'):
            print("Loading from checkpoint")
            checkpoint = torch.load(f'./tmp_{self.job_id.item()}/checkpoint.pth')
            self._epochs_now = checkpoint['epochs_now']
            self._model.module.load_state_dict(checkpoint['model'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._status.load_state(checkpoint['status'])
            self.trainloader.set_batch_size(int(checkpoint['global_bs'] / self._status.accumulation_steps / world_size()))
            self._global_bs = checkpoint['global_bs']

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
    
    def tiaoshi(self):
        print(f"[RANK{rank()}]: time:{time.time()}, mode:{self._status._mode}, lr:{self._optimizer.param_groups[0]['lr']}, global_bs:{self._global_bs}, accu:{self._status.accumulation_steps}, epb_stand:{self._adjuster.now_lower_bound - 1}")

    def beg_epoch(self):
        if self._epochs_now >= self._epochs:
            self.exit()
            return
        if self._beg:
            self._beg = False
            self._status.next_state()
            return
        if self._status.adapt_bs:
            epb_standard = torch.tensor(1.0).to(local_rank())
            if rank() == 0:
                self._ipc.send('standard', '')
                cmd, standard = self._ipc.recv()
                assert cmd == 'standard'
                epb_standard = torch.tensor(float(standard)).to(local_rank())
            dist.broadcast(epb_standard, src=0) # get epb-standard
            self._adjuster.set_epb_standard(epb_standard.item())
            self._global_bs = self._adjuster.adjust_bs(self._grad_monitor.epb, self._global_bs)  # global new bs
            
            contention = torch.tensor(0).to(local_rank())
            if rank() == 0:
                self._ipc.send('contention', '')
                cmd, data = self._ipc.recv()
                assert cmd == 'contention'
                contention = torch.tensor(int(data)).to(local_rank())
            dist.broadcast(contention, src=0) # get contention-level
            
            if contention == 0: # no contention
                if world_size() * self._adjuster.bs_upper >= self._global_bs and world_size() * self._adjuster.bs_lower <= self._global_bs: 
                    # Directly adjust bs without apply new resources
                    if rank() == 0:
                        self._ipc.send('ideal', '')
                    self.trainloader.set_batch_size(int(self._global_bs / world_size()))
                    self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, 1)
                    self._status.accumulation_steps = 1
                elif world_size() * self._adjuster.bs_upper < self._global_bs: # Request more resources
                    new_world_size = math.ceil(self._global_bs * 2 / (self._adjuster.bs_upper + self._adjuster.bs_lower))
                    alloc_result = torch.tensor(0).to(local_rank())
                    if rank() == 0: # Request new resource-allocation
                        self._ipc.send('alloc', new_world_size)
                        cmd, data = self._ipc.recv()
                        assert cmd == 'alloc'
                        alloc_result = torch.tensor(int(data)).to(local_rank())
                    dist.broadcast(alloc_result, src=0)
                    alloc_result = int(alloc_result.item())
                    if alloc_result == world_size():  # Request was rejected, directly accumulate bs on worker
                        accumulation_steps = math.ceil(self._global_bs / world_size() / self._adjuster.bs_upper)
                        self.trainloader.set_batch_size(int(self._global_bs / world_size() / accumulation_steps))
                        self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, accumulation_steps)
                        self._status.accumulation_steps = accumulation_steps
                    else:  # Request was agreed, save checkpoint
                        if rank() == 0:
                            accumulation_steps = math.ceil(self._global_bs / alloc_result / self._adjuster.bs_upper)
                            self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, accumulation_steps)
                            self._status.accumulation_steps = accumulation_steps
                            self.save_checkpoint()
                        dist.barrier()  # Ensure exiting after checkpoint was saved
                        self.exit()
                        return
                else: # Free resource  
                    new_world_size = math.ceil(self._global_bs * 2 / (self._adjuster.bs_upper + self._adjuster.bs_lower))      
                    if new_world_size == world_size():
                        self.trainloader.set_batch_size(int(self._global_bs / world_size()))
                        self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, 1)
                        self._status.accumulation_steps = 1
                    else:
                        if rank() == 0: # Free rebudang resources
                            self._ipc.send('free', new_world_size)
                            self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, 1)
                            self._status.accumulation_steps = 1
                            self.save_checkpoint()
                        dist.barrier()  # Ensure exiting after checkpoint was saved
                        self.exit()
                        return
            elif contention == 1: # contention that not urgent (cannot accumalate)
                new_world_size = math.ceil(self._global_bs / self._adjuster.bs_upper)
                if new_world_size == world_size():  # Directly accumulate bs on worker
                    self.trainloader.set_batch_size(int(self._global_bs / world_size()))
                    self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, 1)
                    self._status.accumulation_steps = 1
                else:
                    if rank() == 0: # Free rebudang resources
                        self._ipc.send('free', new_world_size)
                        self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, 1)
                        self._status.accumulation_steps = 1
                        self.save_checkpoint()
                    dist.barrier()  # Ensure exiting after checkpoint was saved
                    self.exit()
                    return
            elif contention == 2: # contention that not serious (can accumalate)
                if world_size() > 1:
                    new_world_size = world_size() - 1
                    if rank() == 0:
                        self._ipc.send('free', new_world_size)
                        accumulation_steps = math.ceil(self._global_bs / new_world_size / self._adjuster.bs_upper)
                        self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, accumulation_steps)
                        self._status.accumulation_steps = accumulation_steps
                        self.save_checkpoint()
                    dist.barrier()  # Ensure exiting after checkpoint was saved
                    self.exit()
                    return
                else:
                    accumulation_steps = math.ceil(self._global_bs / world_size() / self._adjuster.bs_upper)
                    self.trainloader.set_batch_size(int(self._global_bs / world_size() / accumulation_steps))
                    self._adjuster.adjust_accumulate_step(self._status.accumulation_steps, accumulation_steps)
                    self._status.accumulation_steps = accumulation_steps
                
            self._status.next_state()
            self._adjuster.clear()
            self._grad_monitor.clear()
        elif self._status.adapt_lr:
            self._epochs_now += 1
            self._status.next_state()
            self._adjuster.clear()
            self._grad_monitor.clear()

    def get_epoch(self) -> int:
        return self._epochs_now

    def exit(self):
        if rank() == 0:
            self._ipc.send('end', '')
            self._ipc.close()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(0)
        

