import torch
from elements import _FaaSStatus, _FaaSOptimizer, _FaaSAdjuster, _FaaSGradMonitor, _FaaSLoss, _FaaSModel, FaaSDataLoader

class IntraOptim:

    def __init__(self, model: torch.nn.Module, dataloader: FaaSDataLoader,
                 optimizer: torch.optim.Optimizer, loss_func: torch.nn.Module, accumulation_steps: int):
        self._status = _FaaSStatus(accumulation_steps)
        self._model = _FaaSModel(model)
        self._optimizer = _FaaSOptimizer(optimizer)
        self._loss_func = _FaaSLoss(loss_func)
        self._grad_monitor = _FaaSGradMonitor(model)
        self._adjuster = _FaaSAdjuster()
        self._dataLoader = dataloader
        self._status.associate(self._model)
        self._model.associate(self._status)
        self._optimizer.associate(self._status, self._adjuster)
        self._loss_func.associate(self._status, self._grad_monitor)
        self._adjuster.associate(self._optimizer, self._grad_monitor, self._status, self._dataLoader)
        self._dataLoader.associate(self._status)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss_func(self):
        return self._loss_func


