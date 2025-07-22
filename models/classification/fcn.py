from typing import List

import torch
import torch.nn as nn

from models.base import BaseModel


class OuterLinearLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rank: int = 1, bias: bool = True, cos_coef: float = 80.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.cos_coef = cos_coef

        # Initialize the low-rank matrices
        self.W1 = nn.Parameter(torch.Tensor(in_channels, rank))
        self.W2 = nn.Parameter(torch.Tensor(rank, out_channels))

        if bias: self.bias = nn.Parameter(torch.zeros(out_channels))
        else: self.register_parameter("b", None)

        nn.init.uniform_(self.W1, -torch.pi / 2, torch.pi / 2)
        nn.init.uniform_(self.W2, -torch.pi / 2, torch.pi / 2)

    # check pos embedding like einsum
    def forward(self, x):
        outer_product = torch.einsum("ij,jk->ik", self.W1, self.W2)  # make sure is same
        outer_product = torch.cos(self.cos_coef * outer_product)
        output = x @ outer_product
        if self.bias is not None: output += self.bias
        return output


class LowRankLayer(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, rank: int = 1, **kwargs):
        super().__init__()
        self.oln = OuterLinearLayer(in_channels, out_channels, rank, **kwargs)
        self.activation = nn.SiLU()

    def forward(self, x): return self.activation(self.oln(x))
    def get_alias(self): return "low_rank_layer"
    

class LowRankModel(BaseModel):
    def __init__(self, in_channels: int, hidden_size: List[int], out_channels: int, rank: int = 1, **kwargs):
        super().__init__()
        self.oln1 = LowRankLayer(in_channels, hidden_size[0], rank, **kwargs)
        self.olns = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])
        self.olni = nn.Linear(hidden_size[-1], out_channels)

    def forward(self, x):
        x = self.oln1(x)
        for layer in self.olns: x = layer(x)
        return self.olni(x)

    def get_alias(self): return "low_rank_model"
    