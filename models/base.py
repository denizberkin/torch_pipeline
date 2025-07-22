from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Base class for all models.
    ### Methods:
    - `forward`: Forward pass required by pytorch.
    - `get_alias`: Returns the alias of the model.
    - `get_num_params`: Returns the num of parameters in the model.
    - `get_trainable_params`: Returns the num of trainable parameters in the model.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x): raise NotImplementedError
    def get_alias(self) -> str: return "base"
    def get_num_params(self) -> int: return sum(p.numel() for p in self.parameters())
    def get_trainable_params(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)
