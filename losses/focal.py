import torch
import torch.nn as nn

from losses.base import BaseLoss


class FocalLoss(nn.Module, BaseLoss):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

    def __str__(self):
        return "focal_loss"

    def get_alias(self):
        return "focal_loss"
