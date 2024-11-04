from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import env


class _FaaSStatus:

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._iter_steps = 0
        self._mode = 'init'
        self._model: DistributedDataParallel = None

    def associate(self, model: DistributedDataParallel):
        self._model = model

    def iter_count(self):
        self._iter_steps += 1

    def reset(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._iter_steps = 0

    @property
    def is_update_step(self) -> bool:
        return self._iter_steps % self.accumulation_steps == 0

    @property
    def is_sync_step(self):
        return self._model.no_sync if self.is_update_step else nullcontext

    @property
    def is_accumulation(self):
        return self.accumulation_steps > 1

    @property
    def iter_steps(self) -> int:
        return self._iter_steps

    def set_eval_bs(self):
        self.reset(self.accumulation_steps)
        self._mode = 'eval_bs'

    def set_eval_lr(self):
        self.reset(self.accumulation_steps)
        self._mode = 'eval_lr'

    @property
    def eval_bs(self) -> bool:
        return self._mode == 'eval_bs'

    @property
    def eval_lr(self) -> bool:
        return self._mode == 'eval_lr'


class FaaSDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = DistributedSampler(dataset, num_replicas=env.world_size(), rank=env.rank(), shuffle=shuffle)
        self.status: _FaaSStatus = None
        super().__init__(dataset, batch_size=batch_size, sampler=self.sampler, num_workers=num_workers, **kwargs)

    def associate(self, status: _FaaSStatus):
        self._status = status

    def set_batch_size(self, new_batch_size):
        self._status.set_eval_lr()
        if new_batch_size == self.batch_size:
            if self._iterator is not None:
                self._iterator._reset()
        else:
            self.batch_size = new_batch_size
            self.sampler = DistributedSampler(self.dataset, num_replicas=env.world_size(), rank=env.rank(), shuffle=self.shuffle)
            self.batch_sampler = torch.utils.data.BatchSampler(self.sampler, batch_size=self.batch_size, drop_last=self.drop_last)

    def __iter__(self):
        self._status.set_eval_bs()
        return super().__iter__()


class _FaaSGradMonitor:

    def __init__(self, model: DistributedDataParallel):
        torch.random.seed(42)
        self._model = model

        self.mean = torch.tensor(0.0)
        self.variance = torch.tensor(0.0)
        self.ept_average = torch.tensor(0.0)
        self.iter_steps = 0

        self.gamma = 0.9
        self.grad_max_dim = 500000
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

    def monitor(self):
        self.iter_steps += 1
        para_grad = []
        cur_index = 0
        for param in self._model.parameters():
            num_ele = param.data.numel()
            if self.grad_mask is None:
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

        self.mean = self.gamma * self.mean + (1 - self.gamma) * para_grad
        self.variance = self.gamma * self.variance + (1 - self.gamma) * (para_grad ** 2)

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
        part_size = len(min(self.grad_max_dim, self.grad_dim)) // self.epb_chunks
        sub_mean = torch.split(self.mean, part_size)
        sub_mean = torch.tensor([(chunk ** 2).sum() for chunk in sub_mean])
        sub_variance = torch.split(self.variance, part_size)
        sub_variance = torch.tensor([chunk.sum() for chunk in sub_variance])
        return self.iter_steps / torch.mean(sub_mean / (sub_variance + 1e-30))


class _FaaSAdjuster:

    def __init__(self):
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
        self.convergence_ratio = 0.02
        self.triggle_step = 0
        self.lower_bound = 2
        self.upper_bound = 4
        self.last_epb = torch.tensor(0.0)
        self.now_step = 0
        self.now_triggle = 0

    def associate(self, optimizer: torch.optim.Optimizer, grad_monitor: _FaaSGradMonitor,
                  status: _FaaSStatus, dataloader: FaaSDataLoader):
        self._optimizer = optimizer
        self._grad_monitor = grad_monitor
        self._status = status
        self._dataloader = dataloader

    def clear(self):
        self.last_epb = torch.tensor(0.0)
        self.now_step = 0
        self.now_triggle = 0

    def set_epb(self):
        last_epb = self.last_epb
        self.epb = self._grad_monitor.epb
        self.now_step += 1

        if self.epb - last_epb < last_epb * self.convergence_ratio:
            self.now_triggle += 1
        else:
            self.now_triggle = 0

        if self.now_step >= self.max_step or self.now_triggle >= self.triggle_step:
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
            return int(bs / (self.epb - 1))
        else:
            return 2 * int(bs / (self.epb - 1))


class _FaaSModel(DistributedDataParallel):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model, device_ids=[env.rank()], output_device=env.rank())
        self._status: _FaaSStatus = None

    def associate(self, counter: _FaaSStatus):
        self._status = counter

    def forward(self, *args, **kwargs):
        self._status.iter_count()
        with self._status.is_sync_step:
            tmp = super().forward(*args, **kwargs)
        return tmp


class _FaaSLoss(torch.nn.Module):

    def __init__(self, loss_func: torch.nn.Module):
        self._loss_func = loss_func
        self._status: _FaaSStatus = None
        self._grad_monitor: _FaaSGradMonitor = None

    def associate(self, counter: _FaaSStatus, grad_monitor: _FaaSGradMonitor):
        self._status = counter
        self._grad_monitor = grad_monitor

    def forward(self, *args, **kwargs):
        with self._status.is_sync_step:
            tmp = self._loss_func(*args, **kwargs)
        return tmp

    def backward(self):
        with self._status.is_sync_step:
            self._loss_func.backward()
            self._grad_monitor.monitor()


class _FaaSOptimizer(torch.optim.Optimizer):

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
        self._status: _FaaSStatus = None
        self._adjuster: _FaaSAdjuster = None

    def associate(self, counter: _FaaSStatus, adjuster: _FaaSAdjuster):
        self._status = counter
        self._adjuster = adjuster

    def step(self):
        if self._status.is_update_step:
            if self._status.eval_lr:
                self._optimizer.step()
                self._adjuster.adjust_lr()
            elif self._status.eval_bs:
                self._adjuster.set_epb()
