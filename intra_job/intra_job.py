from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

import env


class Counter:

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._iter_steps = 0
        self._model: DistributedDataParallel = None

    def associate(self, model: DistributedDataParallel):
        self._model = model

    def iter_count(self):
        self._iter_steps += 1

    @property
    def is_update_step(self) -> bool:
        return self._iter_steps % self.accumulation_steps == 0

    def reset(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._iter_steps = 0

    @property
    def is_sync_step(self):
        return self._model.no_sync if self.is_update_step else nullcontext

    @property
    def is_accumulation(self):
        return self.accumulation_steps > 1

    @property
    def iter_steps(self) -> int:
        return self._iter_steps


class GradMonitor:

    def __init__(self, model: DistributedDataParallel):
        torch.random.seed(42)
        self._model = model

        self.mean = torch.tensor(0.0)
        self.variance = torch.tensor(0.0)
        self.ept_average = torch.tensor(0.0)

        self.gamma = 0.9
        self.grad_max_dim = 500000
        self.epb_chunks = 100
        self.iter_steps = 0

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
        self.iter_steps += 1

    def clear(self):
        self.mean = torch.tensor(0.0)
        self.variance = torch.tensor(0.0)
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


class Scheduler:

    def __init__(self, optimizer: torch.optim.Optimizer, grad_monitor: GradMonitor, counter: Counter):
        self._optimizer = optimizer
        self._grad_monitor = grad_monitor
        self._counter = counter

        self.warmup = 30
        self.adjust_interval = 10
        self.low_error = -0.02
        self.high_error = 0.05
        self.eta = 3
        self.standard_ep = 1.3

    def adjust(self):
        if self._counter.iter_steps > self.warmup and self._counter.iter_steps % self.adjust_interval == 0:
            rate = 10 ** ((self.standard_ep - self._grad_monitor.ept) / self.eta)
            if self.low_error <= self._grad_monitor.ept - self.standard_ep <= self.high_error or \
                    torch.isnan(rate) or torch.isinf(rate):
                rate = 1
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= rate


class Model(DistributedDataParallel):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model, device_ids=[env.rank()], output_device=env.rank())
        self._counter: Counter = None

    def associate(self, counter: Counter):
        self._counter = counter

    def forward(self, *args, **kwargs):
        self._counter.iter_count()
        with self._counter.is_sync_step:
            tmp = super().forward(*args, **kwargs)
        return tmp


class Loss(torch.nn.Module):

    def __init__(self, loss_func: torch.nn.Module):
        self._loss_func = loss_func
        self._counter: Counter = None
        self._grad_monitor: GradMonitor = None

    def associate(self, counter: Counter, grad_monitor: GradMonitor):
        self._counter = counter
        self._grad_monitor = grad_monitor

    def forward(self, *args, **kwargs):
        with self._counter.is_sync_step:
            tmp = self._loss_func(*args, **kwargs)
        return tmp

    def backward(self):
        with self._counter.is_sync_step:
            self._loss_func.backward()
            self._grad_monitor.monitor()


class Optimizer(torch.optim.Optimizer):

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
        self._counter: Counter = None
        self._scheduler: Scheduler = None

    def associate(self, counter: Counter, scheduler: Scheduler):
        self._counter = counter
        self._scheduler = scheduler

    def step(self):
        if self._counter.is_sync_step:
            self._optimizer.step()
            self._scheduler.adjust()


class IntraOptim:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 loss_func: torch.nn.Module, accumulation_steps: int):
        self._counter = Counter(accumulation_steps)
        self._model = Model(model)
        self._optimizer = Optimizer(optimizer)
        self._loss_func = Loss(loss_func)
        self._grad_monitor = GradMonitor(model)
        self._scheduler = Scheduler(self._optimizer, self._grad_monitor, self._counter)
        self._counter.associate(self._model)
        self._model.associate(self._counter)
        self._optimizer.associate(self._counter, self._scheduler)
        self._loss_func.associate(self._counter, self._grad_monitor)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss_func(self):
        return self._loss_func
