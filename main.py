import os

from data.build import build_data_loaders
from losses.build import build_loss
from metrics.build import build_metrics
from models.build import build_models
from optimizers.build import build_optimizers
from schedulers.build import build_schedulers
from trainer.build import build_trainer
from utils.config import load_config
from utils.logger import get_logger, log_timeit
from utils.schema import ConfigSchema
from utils.seed import set_seeds

set_seeds(42)


@log_timeit
def main(config: ConfigSchema):
    if config.logging.enabled:
        log_file = config.logging.get("log_file", "train.log")
        log_path = os.path.join("output", config.experiment_name, "logs", log_file)
        logger = get_logger(log_path, config.logging.level)

    losses = build_loss(config.losses)
    metrics = build_metrics(config.metrics)
    trainloader = build_data_loaders(config.data, _type="train")
    valloader = build_data_loaders(config.data, _type="val")
    testloader = build_data_loaders(config.data, _type="test")
    model = build_models(config.model)
    optim = build_optimizers(model.parameters(), config.optim)
    scheduler = build_schedulers(config.optim.scheduler, optimizer=optim)

    trainer = build_trainer(config.train.task)(
        model=model,
        optimizer=optim,
        losses=losses,
        metrics=metrics,
        device=config.device,
        tracker=None,
        scheduler=scheduler,
    )
    trainer.train(
        train_loader=trainloader,
        val_loader=valloader,
        epochs=config.train.epochs,
        log_interval=config.train.log_interval,
        plot=config.train.plot,
        experiment_name=config.experiment_name
    )
    

if __name__ == "__main__":
    conf_folder = os.path.join(os.path.dirname(__file__), "configs")
    config_filenames = [os.path.join(conf_folder, fn) for fn in os.listdir(conf_folder)]
    config: ConfigSchema = load_config(config_filenames, strict=True)

    main(config)
