import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from metrics.base import BaseMetric
from utils.logger import get_logger


class BaseTrainer(ABC):
    """
    Base class for all trainers. 
    A child class is expected to implement `train_epoch`, `validate` and must use `self.compute_loss` logic.
    ### Methods:
    ##### Required to define:
    - `train_epoch`: Logic for one epoch to be implemented.
    - `validate`: Logic for validation to be implemented.
    - `get_alias`: alias to match with config.train.task
    ##### Available:
    - `train`: Train the model for a number of epochs.
    - `compute_loss`: Compute loss and update `loss_history` with current step/batch
    - `average_losses`: Average the losses over the `self.losses_per_epoch`.
    - `_reset_metric_history`: Clear `metric_history`
    - `update_metrics`: Update metrics with current step/batch evals
    - `average_metrics`: Method to average the metrics over the `self.metrics_history`.
    - `__call__`: Method to call the trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        losses: nn.Module,
        metrics: Dict[str, Dict[str, BaseMetric]],
        device: torch.device,
        tracker=None,
        scheduler=None,
        **kwargs,
    ) -> None:
        # optimizer, losses,
        self.model = model.to(device)
        self.optimizer = optimizer
        self.losses = losses
        self.device = torch.device(device)
        self.metrics = metrics
        self.exp_tracker = tracker
        self.scheduler = scheduler
        self.kwargs = kwargs

        # init
        self.metrics_per_epoch = defaultdict(list)
        self.metrics_history = defaultdict(list)  # 
        self.losses_per_epoch = defaultdict(list)
        self.losses_per_step = defaultdict(list)  # track loss per step as well

    @abstractmethod
    def train_epoch(self, dataloader: torch.utils.data.DataLoader): raise NotImplementedError

    @abstractmethod
    def validate(self, dataloader: torch.utils.data.DataLoader): raise NotImplementedError
    def get_alias(self) -> str: return "base"  # must be re-defined by child class

    # Loss related
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for name, loss_data in self.losses.items():
            loss_fn = loss_data["loss_fn"]
            weight = loss_data["weight"]

            loss_weighted = weight * loss_fn(preds, targets)
            total_loss += loss_weighted
            loss_detached = loss_weighted.detach()
            self.losses_per_step[name].append(loss_detached.detach())
        return total_loss

    def average_losses(self) -> None:
        """ should run every epoch end """
        for name, values in self.losses_per_epoch.items():
            self.losses_per_epoch[name].append(sum(values) / len(values))

    # Metrics related
    def _reset_metric_history(self):
        self.metrics_history.clear()
        self.batch_sizes = []

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for single batch, 
        Can be called without the modification on `self.metrics_history` 
        """
        batch_metrics = defaultdict(float)
        for name, metric in self.metrics.items():
            batch_metrics[name] = metric["metric_fn"](preds, targets)
        return batch_metrics

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """ Update `self.metrics_history` based on computed metrics for single batch """
        batch_metrics = self.compute_metrics(preds, targets)
        for name, value in batch_metrics.items():
            self.metrics_history[name].append(value)
        self.batch_sizes.append(batch_size)
        return batch_metrics
    
    def average_metrics(self) -> Dict[str, float]:
        """ should run every epoch end """
        averaged = {}
        for name, values in self.metrics_history.items():
            if len(self.batch_sizes) == len(values):
                metric = sum(val * bs for val, bs in zip(values, self.batch_sizes)) / sum(self.batch_sizes)
            else:
                metric = sum(values) / len(values)
            averaged[name] = metric
            self.metrics_per_epoch[name].append(metric)
        return averaged

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 1,
        log_interval: int = 10,
        plot: bool = False,
        experiment_name: str = "dummy"
    ) -> None:
        logger = get_logger()
        logger.info(f"Starting training for {epochs} epochs.")
        pb = tqdm(range(epochs), desc="Training Epochs", unit="epoch", leave=False)
        for epoch in pb:
            pb.set_description(f"Training epochs {epoch + 1}")
            
            self._reset_metric_history()
            self.train_epoch(train_loader)
            self.average_losses()
            train_metrics = self.average_metrics()
            pb.set_postfix(**train_metrics)
            
            self._reset_metric_history()
            self.validate(val_loader)
            val_metrics = self.average_metrics()
            pb.set_postfix(**val_metrics)

            if self.exp_tracker and epoch % log_interval == 0:
                logger.info(f"Logging metrics for epoch {epoch}.")
                self.exp_tracker.log_epoch(epoch, self.metrics_history)
            
            logger.info(f"Epoch {epoch + 1} - Train Metrics: {train_metrics}")
            logger.info(f"Epoch {epoch + 1} - Validation Metrics: {val_metrics}")
        if plot:
            self.plot_metrics(experiment_name=experiment_name)
        logger.info("Training completed.")

    def plot_metrics(self, experiment_name: str = "dummy") -> None:
        folder = os.path.join("output", experiment_name, "plots")
        if not os.path.exists(folder):
            os.makedirs(folder)

        logger = get_logger()
        logger.info("Plotting metrics...")
        for name, values in self.metrics_per_epoch.items():
            self.plot_stats(xlabel="epochs", ylabel=name + "_epoch", stats=values, dir=folder)
        for name, values in self.losses_per_epoch.items():
            self.plot_stats(xlabel="epochs", ylabel=name + "_epoch", stats=values, dir=folder)
        for name, values in self.losses_per_step.items():
            self.plot_stats(xlabel="steps", ylabel=name + "_step", stats=values, dir=folder)

    def plot_stats(self, xlabel: str, ylabel: str, stats: List[Union[float, int]], dir: str) -> None:
        plt.figure()
        plt.plot(range(len(stats)), stats, label=ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} over epochs")
        plt.legend()
        plt.savefig(os.path.join(dir, f"{ylabel}.png"))
        plt.close()
        

    def __call__(
        self,
        epochs: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        log_interval: int = 10,
    ):
        """ Optionally call the trainer to start training. """
        self.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs, log_interval=log_interval)
