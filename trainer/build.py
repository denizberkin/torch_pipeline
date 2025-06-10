

from trainer.base import BaseTrainer
from utils.utils import find_class_by_alias
from utils.logger import get_logger


def build_trainer(name: str) -> BaseTrainer:
    """
    Build the trainer based on the provided name.
    
    ### Arguments:
        name (str): Name of the trainer to be used.
        
    ### Returns:
        object: Configured trainer instance.
    """
    logger = get_logger()
    trainer_class = find_class_by_alias(name, "trainer")
    
    if trainer_class is None:
        logger.error(f"Trainer class '{name}' not found in 'trainer' directory.")
        raise ValueError(f"Trainer class '{name}' not found in 'trainer' directory.")
    
    logger.info(f"Using trainer '{name}', module name: {__name__}")
    return trainer_class
