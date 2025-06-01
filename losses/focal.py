import torch
import torch.nn as nn

from losses.base import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(self, gamma=2, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.kwargs = kwargs

    def __call__(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

    def get_alias(self):
        return "focal_loss"
