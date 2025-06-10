from abc import ABC, abstractmethod

import torch.optim.lr_scheduler as lrs


class BaseScheduler(lrs.LRScheduler, ABC):
    """ FIXME: Currently WIP, do not use
    Base class for all schedulers.
    A child class is expected to implement `step` and `get_lr` methods.
    """

    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.kwargs = kwargs

    @abstractmethod
    def step(self, epoch: int) -> None: raise NotImplementedError

    @abstractmethod
    def get_lr(self) -> float: raise NotImplementedError