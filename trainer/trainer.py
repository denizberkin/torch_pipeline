from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from trainer.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, losses, metrics, device, tracker=None, config=None, **kwargs) -> None:
        super().__init__(model, optimizer, losses, metrics, device, tracker, **kwargs)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.train()
        for x, y in tqdm(dataloader, desc=f"...", unit="batch", total=len(dataloader)):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outs = self.model(x)
            loss, loss_hist = self.compute_loss(outs, y)
            loss.backward()
            self.optimizer.step()
            self.compute_metrics(outs, y)

    def validate(self, dataloader: torch.utils.data.DataLoader, epoch: int):
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
                x, y = x.to(self.device), y.to(self.device)
                outs = self.model(x)
                outs = torch.argmax(outs, dim=1)  # FIXME: LATEST
                self.compute_metrics(outs, y)

    def compute_loss(self, outs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the given outputs and targets.
        """
        loss_hist = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        total_loss = 0.0
        for name, loss_data in self.losses.items():
            loss_fn = loss_data["loss_fn"]
            weight = loss_data["weight"]
            total_loss += weight * loss_fn(outs, targets)
            loss_hist[name] = weight * loss_fn(outs, targets).detach()
        return total_loss, loss_hist

    def compute_metrics(self, outs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Compute metrics for the given outputs and targets.
        """
        current_metrics = defaultdict(float)
        self.metrics_history["batch_size"].append(outs.shape[0])
        for name, metric_data in self.metrics.items():
            metric_fn = metric_data["metric_fn"]
            metric = metric_fn(
                outs.detach().cpu().view(-1), targets.detach().cpu().view(-1)
            )  # TODO: FIXME: CHECK SHAPE, SIZE, DEVICE, VIEW
            self.metrics_history[name].append(metric)  # each metric will be a list of batches
            current_metrics[name] = metric
        return current_metrics
