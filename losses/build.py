import torch.nn as nn

from utils.constants import LOSSES_DIR
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
    for cfg in cfg_list:
        alias = cfg.name
        weight = cfg.weight if getattr(cfg, "weight", None) else 1.0  # weight is optional
        if alias in PREDEFINED_LOSSES:
            cls = PREDEFINED_LOSSES[alias]
        else:
            cls = find_class_by_alias(alias, LOSSES_DIR)  # losses dir

        if cls is None:
            continue
        logger.info(f"Using class '{alias}' from current module: {cls.__class__.__name__}, module name: {__name__}")
        kwargs = dict(cfg.kwargs) if getattr(cfg, "kwargs", None) else {}
        loss_fn = cls(**kwargs)
        losses[alias] = {"loss_fn": loss_fn, "weight": weight}
    return losses


PREDEFINED_LOSSES = {
    "ce": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,  # bce
    "bce_with_logits": nn.BCEWithLogitsLoss,  # bce + sigmoid
    "mse": nn.MSELoss,
}
