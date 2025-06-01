from typing import List

import torch
import torch.nn as nn

from models.base import BaseModel


class OuterLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 1, bias: bool = True):
        super(OuterLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias

        # Initialize the low-rank matrices
        self.W1 = nn.Parameter(torch.randn(in_features, rank))
        self.W2 = nn.Parameter(torch.randn(rank, out_features))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("b", None)

        nn.init.uniform_(self.W1, -torch.pi / 2, torch.pi / 2)
        nn.init.uniform_(self.W2, -torch.pi / 2, torch.pi / 2)

    def forward(self, x):
        outer_product = torch.einsum("ij,jk->ik", self.W1, self.W2)  # make sure is same
        outer_product = torch.cos(80 * outer_product)

        output = x @ outer_product
        if self.bias is not None:
            output += self.b
        return output


class LowRankModel(BaseModel):
    def __init__(self, in_features: int, hidden_size: List[int], out_features: int, rank: int = 1):
        super(LowRankModel, self).__init__()
        self.oln1 = OuterLinearLayer(in_features, hidden_size[0], rank)

        self.olns = nn.ModuleList(
            [OuterLinearLayer(hidden_size[i], hidden_size[i + 1], rank) for i in range(len(hidden_size) - 1)]
        )
        self.olni = nn.Linear(hidden_size[-1], out_features)

        self.mid_layer_activation = nn.SiLU()
        # self.activation = nn.Tanh()

    def forward(self, x):
        x = self.mid_layer_activation(self.oln1(x))
        for layer in self.olns:
            x = self.mid_layer_activation(layer(x))
        x = self.olni(x)
        return x

    def get_alias(self):
        return "lowrank_model"


class LinearModel(BaseModel):
    def __init__(self, in_features: int, hidden_size: List[int], out_features: int):
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
