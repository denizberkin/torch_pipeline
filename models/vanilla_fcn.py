from typing import List

import torch
import torch.nn as nn

from models.base import BaseModel


class LinearModel(BaseModel):
    def __init__(self, in_features: int, hidden_size: List[int], out_features: int, **kwargs):
        super(LinearModel, self).__init__()
        self.ln1 = nn.Linear(in_features, hidden_size[0])
        self.lns = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])
        self.lni = nn.Linear(hidden_size[-1], out_features)

        self.mid_layer_activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x):
        x = self.mid_layer_activation(self.ln1(x))
        for layer in self.lns:
            x = self.mid_layer_activation(layer(x))
        x = self.lni(x)
        return x

    def get_alias(self):
        return "linear_model"
