from abc import ABC, abstractmethod


class BaseMetric:
    """
    Base class for custom metrics.
    All must implement `get_alias` and `__call__` methods.
    """

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def get_alias(self):
        return "base"
