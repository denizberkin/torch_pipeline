import torch
from tqdm import tqdm

from trainer.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, losses, metrics, device, tracker=None, scheduler=None, **kwargs) -> None:
        super().__init__(model, optimizer, losses, metrics, device, tracker, scheduler, **kwargs)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.train()
        total_loss = 0.0
        pb = tqdm(dataloader, desc="Training...", unit="batch", total=len(dataloader), leave=False)
        for step, (x, y) in enumerate(pb, 1):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            preds = self.model(x)
            loss = self.compute_loss(preds, y)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(preds, dim=1)
            self.update_metrics(preds, y, batch_size=preds.shape[0])
            
            avg_loss = total_loss / step
            pb.set_postfix(loss=f"{avg_loss:.4f}")

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        pb = tqdm(dataloader, desc=f"Validation...", unit="batch", total=len(dataloader), leave=False)
        for x, y in pb:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            preds = torch.argmax(preds, dim=1)
            self.update_metrics(preds, y, preds.shape[0])
    
    def get_alias(self):
        return "classification"