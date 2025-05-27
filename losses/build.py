import torch.nn as nn

from utils.logger import get_logger
from utils.utils import find_class_by_alias


def build_loss(cfg_list) -> dict:
    """
    Build a dictionary of loss functions from loss config
    ### Parameters
        - cfg_list: list of configuration object
    ### Returns
        - losses: dict of loss functions with weights, keys: `loss_fn` and `weight`
    """
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
            cls = find_class_by_alias(alias, "losses")  # losses dir

        if cls is None:
            continue
        logger.info(f"Using class '{alias}' from current module: {cls}, module name: {__name__}")
        kwargs = cfg.kwargs if hasattr(cfg, "kwargs") else {}
        loss_fn = cls(**kwargs)
        losses[alias] = {"loss_fn": loss_fn, "weight": weight}
    return losses
