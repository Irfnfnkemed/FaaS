import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .elements import _FaaSStatus, _FaaSOptimizer, _FaaSAdjuster, _FaaSGradMonitor, FaaSDataLoader
from .env import rank


class IntraOptim:

    def __init__(self, model: torch.nn.Module, trainloader: FaaSDataLoader, testloader: DataLoader,
                 optimizer: torch.optim.Optimizer, accumulation_steps: int):
        self._status = _FaaSStatus(accumulation_steps)
        self._model = DDP(model, device_ids=[rank()])
        self._optimizer = _FaaSOptimizer(optimizer)
        self._grad_monitor = _FaaSGradMonitor(model)
        self._adjuster = _FaaSAdjuster()
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs_now = 0
        self._status.associate(self._model)
        self._optimizer.associate(self._status, self._adjuster, self._grad_monitor)
        self._adjuster.associate(self._status, self._optimizer, self._grad_monitor, self.trainloader)
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
            self._status.set_adapt_lr()
            self._grad_monitor.clear()
            self.epochs_now += 1
        elif self._status.adapt_lr:
            self._status.set_adapt_bs()
            self._grad_monitor.clear()
