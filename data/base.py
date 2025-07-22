from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    ### Base Dataset class
    All datasets inheriting must implement: `load_data`, `get_alias`, `__getitem__`, and `__len__` methods.
    ### Methods:
        - `load_data` (required): Load the dataset.
        - `get_alias` (required): return dataset alias to match with config.
        - `__getitem__` (required): indexing mechanism, expected by pytorch.
        - `__len__` (required): return dataset length, expected by pytorch.
        - `collate_fn` (optional): Method to customize how batches are collated.
    """
    def __init__(self, data_dir: str, _type: str):
        super().__init__()
        self.data_dir = data_dir
        self._type = _type  # train, val or test

    @abstractmethod
    def load_data(self): raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index): raise NotImplementedError

    @abstractmethod
    def __len__(self): raise NotImplementedError
    def get_alias(self): return "base"
    def collate_fn(self, batch): return batch
