""" 
This contains the complex activation functions, all are nn.Module subclasses.
"""

import torch
import torch.nn as nn


class ComplexSoftmax(nn.Module):
    """ returns complex """
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.log(1 + torch.exp(x))


class ComplexTanh(nn.Module):
    """ returns complex """
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class ComplexLinear(nn.Module):
    """ returns complex """
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            1 / (1 + torch.exp(-x.real)) + 
            torch.complex(torch.tensor(0.0), torch.tensor(1.0)) / (1 + torch.exp(-x.imag))
            )
    

class ComplexCardioid(nn.Module):
    """ returns complex """
    def __init__(self): 
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.c.device = x.device  # ensure we're on same device pls
        return self.c * (1 + x.real / torch.sqrt(x.real**2 + x.imag**2)) * x
    

class ComplexSpline(nn.Module):
    """ returns complex """
    def __init__(self):
        super().__init__()
        self.pow = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.pow.device = x.device
        return x**self.pow * torch.log(x)
    

class ComplexSilu(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.alpha.device = x.device
        return x.real / (1.0 + torch.exp(-torch.complex(self.alpha * x.real, torch.cos(x.imag))))
    

class ComplexSigmoid(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return 1.0 / (1.0 + torch.exp(-x))


class ComplexRootNorm(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.sqrt(x.real**2 + x.imag**2)


class ComplexNorm(nn.Module):
    def __init__(self): """ ### Returns real num !!! """; super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x.real**2 + x.imag**2


# weird hacks??
class ComplexNorm2(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            torch.tensor(torch.tanh(x.real) / (1 - (x.real - 3) * torch.exp(-x.real))),
            torch.tensor(torch.tanh(x.imag) / (1 - (x.imag - 3) * torch.exp(-x.imag)))
        )
