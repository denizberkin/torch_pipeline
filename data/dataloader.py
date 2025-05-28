from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader, ABC):
    """
    Base class for data loaders, extending torch's DataLoader.
    This class can be extended to implement custom data loading logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def load_data(self):
        """
        Abstract method to load data.
        This method should be implemented by subclasses to define how data is loaded.
        """
        pass
