from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """
    ### Base Loss class
    All must implement `get_alias` and `__call__` methods.
    ### Methods:
        - `get_alias` (required): return loss alias to match with config.
        - `__call__` (required): compute and return loss given targets and preds.
    """

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def get_alias(self):
        return "base"
