from typing import List

import torch
import torch.nn as nn

from models.base import BaseModel


class OuterLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 1, bias: bool = True, cos_coefficient: float = 80.0):
        super(OuterLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias
        self.cos_coefficient = cos_coefficient

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
        outer_product = torch.cos(self.cos_coefficient * outer_product)

        output = x @ outer_product
        if self.bias is not None:
            output += self.b
        return output


class LowRankLayer(BaseModel):
    def __init__(self, in_features: int, out_features: int, rank: int = 1, **kwargs):
        super(LowRankLayer, self).__init__()
        self.oln = OuterLinearLayer(in_features, out_features, rank)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.oln(x))
        return x

    def get_alias(self):
        return "low_rank_layer"
    

class LowRankModel(BaseModel):
    def __init__(self, in_features: int, hidden_size: List[int], out_features: int, rank: int = 1, **kwargs):
        super(LowRankModel, self).__init__()
        self.oln1 = OuterLinearLayer(in_features, hidden_size[0], rank)
        self.olns = nn.ModuleList([OuterLinearLayer(hidden_size[i], hidden_size[i + 1], rank) for i in range(len(hidden_size) - 1)])
        self.olni = nn.Linear(hidden_size[-1], out_features)

        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.activation(self.oln1(x))
        for layer in self.olns:
            x = self.activation(layer(x))
        x = self.olni(x)
        return x

    def get_alias(self):
        return "low_rank_model"