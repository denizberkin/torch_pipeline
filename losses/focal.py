import torch
from torch.nn.functional import cross_entropy

from losses.base import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(self, gamma=2, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.kwargs = kwargs

    def __call__(self, inputs, targets):
        ce_loss = cross_entropy(inputs, targets, reduction="none")
        return ((1 - torch.exp(-ce_loss))**self.gamma * ce_loss).mean()

    def get_alias(self): return "focal_loss"
