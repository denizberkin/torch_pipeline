import os

from data.build import build_data_loaders
from losses.build import build_loss
from metrics.build import build_metrics
from trainer.trainer import ClassificationTrainer
from utils.config import load_config
from utils.logger import get_logger


def main():
    conf_folder = os.path.join(os.path.dirname(__file__), "configs")
    config_filenames = [os.path.join(conf_folder, fn) for fn in os.listdir(conf_folder)]
    config = load_config(config_filenames, strict=True)
    if config.logging.enabled:
        log_path = os.path.join(config.logging.log_dir, "exp.log")
        logger = get_logger(log_path, config.logging.level)

    losses = build_loss(config.losses)
    metrics = build_metrics(config.metrics)
    dataloader = build_data_loaders(config.dataloader)
    ClassificationTrainer(
        model=None,
        optimizer=None,
        losses=losses,
        metrics=metrics,
        device=config.device,
        tracker=None,
        config=config,
    )


if __name__ == "__main__":
    main()
