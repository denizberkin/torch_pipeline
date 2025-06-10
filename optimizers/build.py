from typing import Iterator, Union

import torch
import torch.optim as optim

from utils.schema import OptimConfig
from utils.logger import get_logger


def build_optimizers(
    model_params: Iterator[torch.Tensor],
    config: OptimConfig,
) -> optim.Optimizer:
    """
    Build optimizers based on the provided configuration.
    ### Parameters
        - config: Configuration dictionary containing optimizer settings.
    ### Returns
        - An instance of the optimizer.
    """
    logger = get_logger()
    optim_class = PREDEFINED_OPTIMIZERS.get(config.name, None)
    if optim_class is None:
        logger.error(f"Optimizer class '{config.name}' not found in predefined optimizers. No custom optim yet.")
        raise ValueError(f"Optimizer class '{config.name}' not found in predefined optimizers. No custom optim yet.")
    logger.info(f"Using class '{config.name}', module name: {__name__}")
    return optim_class(model_params, config.lr, **config.kwargs)


PREDEFINED_OPTIMIZERS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}
