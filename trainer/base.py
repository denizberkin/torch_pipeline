from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from metrics.base import BaseMetric
from utils.logger import get_logger


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        losses: nn.Module,
        metrics: Dict[str, BaseMetric],
        device: torch.device,
        tracker=None,
        **kwargs,
    ) -> None:
        # optimizer, losses,
        self.model = model.to(device)
        self.optimizer = optimizer
        self.losses = losses
        self.device = device
        self.metrics = metrics
        self.exp_tracker = tracker
        self.metrics_history = defaultdict(list)
        self.kwargs = kwargs

    @abstractmethod
    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        pass

    @abstractmethod
    def validate(self, dataloader: torch.utils.data.DataLoader):
        pass

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 1,
        log_interval: int = 10,
    ) -> None:
        logger = get_logger()
        logger.info(f"Starting training for {epochs} epochs.")
        for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch", leave=False):
            self.train_epoch(train_loader)
            if val_loader:
                self.validate(val_loader)
            if self.exp_tracker and epoch % log_interval == 0:
                logger.info(f"Logging metrics for epoch {epoch}.")
                self.exp_tracker.log_epoch(epoch, self.metrics_history)
            self.metrics_history.clear()
        logger.info("Training completed.")
