from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from . import env


class _FaaSStatus:

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._iter_steps = 0
        self._break = False
        self._mode = 'adapt_lr'
        self._model: DistributedDataParallel = None

    def associate(self, model: DistributedDataParallel):
        self._model = model

    def iter_count(self):
        self._iter_steps += 1

    def reset(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._iter_steps = 0
        self._break = False

    @property
    def is_update_step(self) -> bool:
        return self._iter_steps % self.accumulation_steps == 0

    @property
    def sync_or_not(self):
        return self._model.no_sync() if self.is_update_step else nullcontext

    @property
    def is_accumulation(self) -> bool:
        return self.accumulation_steps > 1

    @property
    def iter_steps(self) -> int:
        return self._iter_steps

    def set_adapt_bs(self):
        self.reset(self.accumulation_steps)
        self._mode = 'adapt_bs'
        self._model.eval()

    def set_adapt_lr(self):
        self.reset(self.accumulation_steps)
        self._mode = 'adapt_lr'
        self._model.train()

    def set_break(self):
        self._break = True

    @property
    def adapt_bs(self) -> bool:
        return self._mode == 'adapt_bs'

    @property
    def adapt_lr(self) -> bool:
        return self._mode == 'adapt_lr'

    @property
    def is_break(self) -> bool:
        return self._break


class FaaSDataLoader:
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._kwargs = kwargs
        self._sampler = DistributedSampler(dataset, num_replicas=env.world_size(), rank=env.rank(), shuffle=shuffle)
        self._status: _FaaSStatus = None
        self._dataloader = DataLoader(dataset, batch_size=batch_size, sampler=self._sampler, num_workers=num_workers, **kwargs)

    def __iter__(self):
        return self._dataloader.__iter__()

    def __getattr__(self, name):
        attr = getattr(self._dataloader, name, None)
        if attr is None:
            raise AttributeError(f"'FaaSDataLoader' object has no attribute '{name}'")
        return attr

    def associate(self, status: _FaaSStatus):
        self._status = status

    def set_batch_size(self, new_batch_size):
        if new_batch_size != self._batch_size:
            self._batch_size = new_batch_size
            self._sampler = DistributedSampler(self._dataset, num_replicas=env.world_size(), rank=env.rank(), shuffle=self._shuffle)
            self._dataloader = DataLoader(self._dataset, batch_size=new_batch_size, sampler=self._sampler,
                                          num_workers=self._num_workers, **self._kwargs)
        self._status.set_break()


class _FaaSGradMonitor:

    def __init__(self, model: DistributedDataParallel):
        self._status: _FaaSStatus = None
        self._model = model

        self.mean = torch.tensor(0.0)
        self.variance = torch.tensor(0.0)
        self.ept_average = torch.tensor(0.0)
        self.iter_steps = 0

        self.gamma = 0.9
        self.grad_max_dim = 50000000
        self.epb_chunks = 100

        self.grad_dim = 0
        for param in self._model.parameters():
            self.grad_dim += param.data.numel()

        if self.grad_dim <= self.grad_max_dim:
            self.grad_mask = None
        else:
            self.grad_mask = torch.zeros(self.grad_dim, dtype=torch.bool)
            sample_indices = torch.randint(0, self.grad_dim, (self.grad_max_dim,))
            self.grad_mask[sample_indices] = True

    def associate(self, status: _FaaSStatus):
        self._status = status

    def monitor(self):
        self.iter_steps += 1
        para_grad = []
        cur_index = 0
        for param in self._model.parameters():
            num_ele = param.data.numel()
            if self.grad_mask is not None:
                sub_mask = self.grad_mask[cur_index: cur_index + num_ele]
                if param.grad is not None:
                    para_grad.append(param.grad[sub_mask].detach().view(-1))
                else:
                    para_grad.append(torch.zeros(torch.sum(sub_mask)), param.data.dtype)
            else:
                if param.grad is not None:
                    para_grad.append(param.grad.detach().view(-1))
                else:
                    para_grad.append(num_ele, param.data.dtype)
            cur_index += num_ele
        para_grad = torch.cat(para_grad)

        if self._status.adapt_lr:
            self.mean = self.gamma * self.mean + (1 - self.gamma) * para_grad
            self.variance = self.gamma * self.variance + (1 - self.gamma) * (para_grad ** 2)
        elif self._status.adapt_bs:
            self.mean = self.mean + para_grad
            self.variance = self.variance + para_grad ** 2

    def clear(self):
        self.mean = torch.tensor(0.0)
        self.variance = torch.tensor(0.0)
        self.ept_average = torch.tensor(0.0)
        self.iter_steps = 0

    @property
    def ept(self) -> torch.Tensor:
        ratio = self.mean ** 2 / (self.variance + 1e-30)
        ratio = ratio[ratio < 1]
        ept = torch.log10(1 / torch.mean(ratio))
        if torch.isnan(ept) or torch.isinf(ept):
            ept = torch.tensor(0.0)
        self.ept_average = self.ept_average * 0.8 + ept * 0.2
        return self.ept_average

    @property
    def epb(self) -> torch.Tensor:
        part_size = min(self.grad_max_dim, self.grad_dim) // self.epb_chunks
        sub_mean = torch.split(self.mean, part_size)
        sub_mean = torch.tensor([(chunk ** 2).sum() for chunk in sub_mean])
        sub_variance = torch.split(self.variance, part_size)
        sub_variance = torch.tensor([chunk.sum() for chunk in sub_variance])
        return self.iter_steps / torch.mean(sub_mean / (sub_variance + 1e-30))


class _FaaSAdjuster:

    def __init__(self):
        self.epb: torch.Tensor = None
        self._optimizer: torch.optim.Optimizer = None
        self._grad_monitor: _FaaSGradMonitor = None
        self._status: _FaaSStatus = None
        self._dataloader: FaaSDataLoader = None

        # config of ept
        self.warmup_ept = 30
        self.adjust_interval = 10
        self.low_error = -0.02
        self.high_error = 0.05
        self.eta = 3
        self.standard_ept = 1.3

        # config of epb
        self.max_step = 50
        self.convergence_ratio = 0.05
        self.trigger_step = 5
        self.lower_bound = 2
        self.upper_bound = 4
        self.last_epb = torch.tensor(0.0)
        self.now_step = 0
        self.now_trigger = 0

    def associate(self, status: _FaaSStatus, optimizer: torch.optim.Optimizer,
                  grad_monitor: _FaaSGradMonitor, dataloader: FaaSDataLoader):
        self._status = status
        self._optimizer = optimizer
        self._grad_monitor = grad_monitor
        self._dataloader = dataloader

    def clear(self):
        self.last_epb = torch.tensor(0.0)
        self.now_step = 0
        self.now_trigger = 0

    def set_epb(self):
        last_epb = self.last_epb
        self.epb = self._grad_monitor.epb
        self.now_step += 1

        if self.epb - last_epb < last_epb * self.convergence_ratio:
            self.now_trigger += 1
        else:
            self.now_trigger = 0

        if self.now_step >= self.max_step or self.now_trigger >= self.trigger_step:
            self._grad_monitor.clear()
            self._dataloader.set_batch_size(self.adjust_bs(self._dataloader.batch_size))
            self.clear()

    def adjust_lr(self):
        if self._status.iter_steps > self.warmup_ept and self._status.iter_steps % self.adjust_interval == 0:
            rate = 10 ** ((self.standard_ept - self._grad_monitor.ept) / self.eta)
            if self.low_error <= self._grad_monitor.ept - self.standard_ept <= self.high_error or \
                    torch.isnan(rate) or torch.isinf(rate):
                rate = 1
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= rate

    def adjust_bs(self, bs: int) -> int:
        if self.lower_bound <= self.epb <= self.upper_bound:
            return bs
        elif self.epb < self.lower_bound:
            return int(0.5 * bs * self.epb)
        else:
            return int(bs * (1 + self.epb - self.upper_bound))


class _FaaSOptimizer:

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._grad_monitor: _FaaSGradMonitor = None
        self._optimizer = optimizer
        self._status: _FaaSStatus = None
        self._adjuster: _FaaSAdjuster = None

    def __getattr__(self, name):
        attr = getattr(self._optimizer, name, None)
        if attr is None:
            raise AttributeError(f"'_FaaSOptimizer' object has no attribute '{name}'")
        return attr

    def associate(self, counter: _FaaSStatus, adjuster: _FaaSAdjuster, grad_monitor: _FaaSGradMonitor):
        self._status = counter
        self._grad_monitor = grad_monitor
        self._adjuster = adjuster

    def step(self):
        self._status.iter_count()
        if self._status.is_update_step:
            self._grad_monitor.monitor()
            if self._status.adapt_lr:
                self._optimizer.step()
                self._adjuster.adjust_lr()
            elif self._status.adapt_bs:
                self._adjuster.set_epb()