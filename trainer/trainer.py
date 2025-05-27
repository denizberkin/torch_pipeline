import torch
import torch.nn as nn
from tqdm import tqdm

from trainer.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, losses, metrics, device, tracker=None, **kwargs) -> None:
        """
        self.losses
        """
        super().__init__(model, optimizer, losses, metrics, device, tracker, **kwargs)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int):
        self.model.train()
        for x, y in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outs = self.model(x)
            loss = self.compute_loss(outs, y)
            loss.backward()
            self.optimizer.step()
            self.compute_metrics(outs, y)

    def compute_loss(self, outs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the given outputs and targets.
        """
        loss = 0.0
        for name, loss_data in self.losses.items():
            loss_fn = loss_data["loss_fn"]
            weight = loss_data["weight"]
            loss += weight * loss_fn(outs, targets)
        return loss

    def compute_metrics(self, outs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Compute metrics for the given outputs and targets.
        """
        for name, metric_data in self.metrics:
            pass  # TODO:
