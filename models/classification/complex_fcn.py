import torch
import torch.nn as nn

from models.classification.activations import *
from models.base import BaseModel


class ComplexOuterLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 1,
                 bias: bool = True, cos_coef: float = 80.0, sin_coef: float = 80.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.cos_coef = cos_coef
        self.sin_coef = sin_coef

        if bias: self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
        else: self.register_parameter("bias", None)
        
        # Initialize low-rank matrices
        scale = torch.sqrt(torch.tensor(torch.pi))
        self.W0 = ...  # this is defined in original but not used?
        self.W1 = nn.Parameter(torch.complex(
            (torch.randn(in_features, rank) - 0.5) * 2 * scale,
            (torch.randn(in_features, rank) - 0.5) * 2 * scale
        ))
        self.W2 = nn.Parameter(torch.complex(
            (torch.randn(rank, out_features) - 0.5) * 2 * scale,
            (torch.randn(rank, out_features) - 0.5) * 2 * scale
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer_product = torch.einsum("ij,jk->ik", self.W1, self.W2)
        j = torch.complex(torch.tensor([0.0]), torch.tensor([1.0]))  # 0 + j
        outer_product = self.cos_coef * torch.exp(j * self.sin_coef * outer_product.real)
        output = x.to(torch.complex64) @ outer_product
        if self.bias is not None: output += self.bias
        return output


class ComplexLowRankLayer(BaseModel):
    def __init__(self, in_features: int, out_features: int, rank: int = 1, **kwargs):
        super().__init__()
        self.coln = ComplexOuterLinearLayer(in_features, out_features, rank)
        self.activation = ComplexNorm()  # choose your activation

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.activation(self.coln(x))
    def get_alias(self): return "complex_low_rank_layer"


class ComplexLowRankModel(BaseModel):
    def __init__(self, in_features: int, hidden_size: list[int], out_features: int, rank: int = 1, **kwargs):
        super().__init__()
        self.coln1 = ComplexLowRankLayer(in_features, hidden_size[0], rank, **kwargs)
        self.colns = nn.ModuleList([ComplexLowRankLayer(hidden_size[i], hidden_size[i + 1], rank, **kwargs) for i in range(len(hidden_size) - 1)])
        self.colni = nn.Linear(hidden_size[-1], out_features, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coln1(x)
        for layer in self.colns: x = layer(x)
        return self.colni(x)
    
    def get_alias(self): return "complex_low_rank_model"



""" TINYGRAD MNIST MODEL DEF

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = nn.Conv2d(1, 32, 5)
    self.c2 = nn.Conv2d(32, 32, 5)
    self.bn1 = nn.BatchNorm2d(32)
    self.m1 = nn.MaxPool2d(2)
    self.c3 = nn.Conv2d(32, 64, 3)
    self.c4 = nn.Conv2d(64, 64, 3)
    self.bn2 = nn.BatchNorm2d(64)
    self.m2 = nn.MaxPool2d(2)
    self.lin = nn.Linear(576, 10)
  def forward(self, x):
    x = nn.functional.relu(self.c1(x))
    x = nn.functional.relu(self.c2(x), 0)
    x = self.m1(self.bn1(x))
    x = nn.functional.relu(self.c3(x), 0)
    x = nn.functional.relu(self.c4(x), 0)
    x = self.m2(self.bn2(x))
    return self.lin(torch.flatten(x, 1))

"""