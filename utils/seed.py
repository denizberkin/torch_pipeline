import random

import torch
import numpy as np


def set_seeds(seed: int = 42) -> None:
    """
    Set random seed for reproducibility* in experiments.
    
    *Note: This does not guarantee full reproducibility due to non-deterministic nature of nn's
    It will, however, be consistent within a small range of variation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # it is ignored if CUDA is not available
    
    # probably drops performance, but ensures ~deterministic behaviour on low-level ops, uncomment to use
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
