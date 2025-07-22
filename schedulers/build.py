"""TODO: add some custom or interesting schedulers"""
from typing import List

import torch

from utils.logger import get_logger
from utils.schema import SchedulerConfig


def build_schedulers(config: SchedulerConfig, 
                    optimizers: torch.optim.Optimizer | List[torch.optim.Optimizer]
                    ) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build a learning rate scheduler based on the provided configuration.
    
    ### Arguments:
        scheduler_name (str): Name of the scheduler to be used.
        scheduler_config (dict): Configuration dictionary for the scheduler.
        optimizer (torch.optim.Optimizer): Optimizer for which the scheduler will be applied.
    ### Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler instance(s)
        depending on number of optimizers provided.
    """
    logger = get_logger()
    try: scheduler_class = PREDEFINED_SCHEDULERS.get(config.name, None)
    except AttributeError: logger.warning("scheduler.name is null, passing"); return None
    if scheduler_class is None:
        logger.error(f"Scheduler class '{config.name}' not found in 'schedulers' directory.")
        raise ValueError(f"Scheduler class '{config.name}' not found in 'schedulers' directory.")
    logger.info(f"Using scheduler '{config.name}', module name: {__name__}")
    kwargs = dict(config.kwargs) if getattr(config, "kwargs", None) else {}
    if isinstance(optimizers, list): return [scheduler_class(opt) for opt in optimizers]
    return scheduler_class(optimizers, **kwargs)


PREDEFINED_SCHEDULERS = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "multi_step_lr": torch.optim.lr_scheduler.MultiStepLR,
    "exponential_lr": torch.optim.lr_scheduler.ExponentialLR,
    "cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR,
    "cyclic_lr": torch.optim.lr_scheduler.CyclicLR,
}