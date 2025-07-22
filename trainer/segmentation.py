import torch
from tqdm import tqdm

from trainer.base import BaseTrainer
from utils.logger import get_logger


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, losses, metrics, device, tracker=None, scheduler=None, **kwargs) -> None:
        super().__init__(model, optimizer, losses, metrics, device, tracker, scheduler, **kwargs)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.train()
        total_loss = 0.0
        pb = tqdm(dataloader, desc="Training...", unit="batch", total=len(dataloader), leave=False)
        for step, batch in enumerate(pb, 1):
            pass

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        pb = tqdm(dataloader, desc="Validating...", unit="batch", total=len(dataloader), leave=False)
        for batch in pb:
            pass

    @torch.no_grad()
    def test(self, dataloader: torch.utils.data.DataLoader):
        self._reset_metric_history()
        self.model.eval()
        pb = tqdm(dataloader, desc="Testing...", unit="batch", total=len(dataloader), leave=False)
        for batch in pb:
            pass
        return
    
    def get_alias(self):
        return "segmentation"
