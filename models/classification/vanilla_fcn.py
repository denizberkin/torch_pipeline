from typing import List

import torch.nn as nn

from models.base import BaseModel


class LinearModel(BaseModel):
    def __init__(self, in_channels: int, hidden_size: List[int], out_channels: int, **kwargs):
        super().__init__()
        self.ln1 = nn.Linear(in_channels, hidden_size[0])
        self.lns = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])
        self.lni = nn.Linear(hidden_size[-1], out_channels)

        self.mid_layer_activation = nn.SiLU()

    def forward(self, x):
        x = self.mid_layer_activation(self.ln1(x))
        for layer in self.lns: x = self.mid_layer_activation(layer(x))
        return self.lni(x)
    
    def get_alias(self): return "linear_model"
