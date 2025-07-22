"""TODO: add more optimizers, predefined and/or custom"""
from typing import List

import torch.nn as nn
import torch.optim as optim

from utils.schema import OptimConfig
from utils.logger import get_logger


def build_optimizers(
    models: nn.Module | List[nn.Module],
    config: OptimConfig,
) -> optim.Optimizer | List[optim.Optimizer]:
    """
    Build optimizers based on the provided configuration and model(s).
    ### Parameters:
        - config: Configuration dictionary containing optimizer settings.
    ### Returns:
        Instance(s) of the optimizer depending on the number of models provided.
    """
    logger = get_logger()
    optim_class = PREDEFINED_OPTIMIZERS.get(config.name, None)
    if optim_class is None:
        logger.error(f"Optimizer class '{config.name}' not found in predefined optimizers. No custom optim yet.")
        raise ValueError(f"Optimizer class '{config.name}' not found in predefined optimizers. No custom optim yet.")
    logger.info(f"Using class '{config.name}', module name: {__name__}")
    if isinstance(models, list): return [optim_class(m.parameters()) for m in models]
    return optim_class(models.parameters(), config.lr, **config.kwargs)


PREDEFINED_OPTIMIZERS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}
