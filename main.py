import os
import argparse
from typing import List

from data.build import build_data_loaders
from losses.build import build_loss
from metrics.build import build_metrics
from models.build import build_models
from optimizers.build import build_optimizers
from schedulers.build import build_schedulers
from trainer.base import BaseTrainer
from trainer.build import build_trainer
from utils.config import load_config
from utils.logger import get_logger, log_timeit
from utils.schema import ConfigSchema
from utils.seed import set_seeds

set_seeds(1337)


@log_timeit
def main(config: ConfigSchema):
    if config.logging.enabled:
        log_file = config.logging.get("log_file", "train.log")
        log_path = os.path.join("output", config.experiment_name, "logs", log_file)
        logger = get_logger(log_path, config.logging.level)

    losses = build_loss(config.losses)
    metrics = build_metrics(config.metrics)
    models = build_models(config.model)
    optims = build_optimizers(models, config.optim)
    schedulers = build_schedulers(config.optim.scheduler, optimizers=optims)
    # build dataloaders last, as they are the slowest
    trainloader = build_data_loaders(config.data, _type="train")
    valloader = build_data_loaders(config.data, _type="val")
    testloader = build_data_loaders(config.data, _type="test")
    logger.info([str(m) for m in models])

    trainers: List[BaseTrainer] = [build_trainer(config.train.task)(
        model=models[i], optimizer=optims[i],
        losses=losses, metrics=metrics,
        device=config.device, tracker=None,
        scheduler=schedulers)
        for i in range(len(models))]
    
    for trainer in trainers:
        trainer.train(
            train_loader=trainloader,
            val_loader=valloader,
            epochs=config.train.epochs,
            log_interval=config.train.log_interval,
            plot=config.train.plot,
            experiment_name=config.experiment_name
        )
        trainer.test(testloader)
        trainer.save_model(folder=os.path.join("output", config.experiment_name, "models"), 
                           filename=f"{trainer.model.get_alias()}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="path to yaml within configs/ folder")
    args = parser.parse_args()

    try:  # either provide path or filename within configs folder 
       config: ConfigSchema = load_config(args.cfg, strict=True)
    except FileNotFoundError as e:
        conf_folder = os.path.join(os.path.dirname(__file__), "configs")
        config_fn = os.path.join(conf_folder, args.cfg)
        config: ConfigSchema = load_config(config_fn, strict=True)

    main(config)
