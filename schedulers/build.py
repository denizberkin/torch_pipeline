
import torch

from utils.logger import get_logger
from utils.schema import SchedulerConfig


def build_schedulers(config: SchedulerConfig, 
                    optimizer: torch.optim.Optimizer
                    ) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a learning rate scheduler based on the provided configuration.
    
    ### Arguments:
        scheduler_name (str): Name of the scheduler to be used.
        scheduler_config (dict): Configuration dictionary for the scheduler.
        optimizer (torch.optim.Optimizer): Optimizer for which the scheduler will be applied.
    ### Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler instance.
    """

    logger = get_logger()

    scheduler_class = PREDEFINED_SCHEDULERS.get(config.name, None)
    if scheduler_class is None:
        logger.error(f"Scheduler class '{config.name}' not found in 'schedulers' directory.")
        raise ValueError(f"Scheduler class '{config.name}' not found in 'schedulers' directory.")
    logger.info(f"Using scheduler '{config.name}', module name: {__name__}")
    kwargs = dict(config.kwargs) if getattr(config, "kwargs", None) else {}
    scheduler = scheduler_class(optimizer, **kwargs)
    return scheduler


PREDEFINED_SCHEDULERS = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "multi_step_lr": torch.optim.lr_scheduler.MultiStepLR,
    "exponential_lr": torch.optim.lr_scheduler.ExponentialLR,
    "cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR,
    "cyclic_lr": torch.optim.lr_scheduler.CyclicLR,
}