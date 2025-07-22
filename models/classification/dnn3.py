
import torch
import torch.nn as nn

from models.classification.fcn import OuterLinearLayer
from models.base import BaseModel


def _weights_init_mnist(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class DELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.n, 0.0)

    def forward(self, x):
        left = (x <= 0.0) * nn.functional.silu(x)
        delu = (self.n + 0.5) * x + torch.abs(torch.exp(-x) - 1.0)
        right = (x > 0.0) * delu
        return left + right


class DNN(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        in_channels = 784
        self.layers = nn.Sequential(
        nn.Linear(in_channels, 392),
        # DELU(),
        nn.ReLU(),
        nn.Linear(392, 196),
        # DELU(),
        nn.ReLU(),
        nn.Linear(196, 10),
        )

        self.apply(_weights_init_mnist)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_alias(self):
        return "dnn3"


class DNN(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        in_channels = 784

        self.layers = nn.Sequential(
        # nn.Linear(in_channels, 392),
        OuterLinearLayer(in_channels, 392, kwargs.get("rank", 3)),
        # DELU(),
        nn.ReLU(),
        nn.Linear(392, 196),
        # OuterLinearLayer(392, 196, kwargs.get("rank", 3)),
        # DELU(),
        nn.ReLU(),
        nn.Linear(196, 10),
        )

        self.apply(_weights_init_mnist)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_alias(self):
        return "dnn3_lowrank"