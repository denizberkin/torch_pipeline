from typing import Iterator, Union

import torch
import torch.optim as optim

from utils.schema import OptimConfig


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
    return optim.Adam(model_params, config.lr, **config.kwargs)
