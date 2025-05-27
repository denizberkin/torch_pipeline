import torch.nn as nn

from utils.logger import get_logger


def build_loss(cfg_list):
    logger = get_logger()
    losses = {}
    predefined_losses = {
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "bce": nn.BCELoss,
    }
    for cfg in cfg_list:
        alias = cfg.name
        weight = cfg.weight if hasattr(cfg, "weight") else 1.0

        if alias in predefined_losses:
            cls = predefined_losses[alias]
        else:
            # check class name in the current module
            pass
            # cls = get_class_by_str(sys.modules, alias)  # FIXME: fix logic of this

        if cls is None:
            continue
        logger.info(f"Using class '{alias}' from current module: {cls}, module name: {__name__}")

        kwargs = {k: v for k, v in cfg.items() if k != "name" and k != "weight"}
        loss_fn = cls(**kwargs)
        losses[alias] = {
            "loss_fn": loss_fn,
            "weight": weight,
            "kwargs": kwargs,
        }
    return losses
