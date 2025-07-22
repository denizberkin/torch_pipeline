""" keep track of utility performance metrics """
import time
from typing import Dict

import torch
import torch.nn as nn

@torch.no_grad()
def forward_pass_time(model: nn.Module, inputs: torch.Tensor, return_seconds: bool=False) -> float:
    start = time.perf_counter()
    _ = model(inputs)
    end = time.perf_counter()
    if return_seconds: return 1000 * (end - start) 
    return end - start


def memory_usage() -> Dict[str, float]:
    if torch.cuda.is_available():
        return {
            "gpu_allocated_mb": torch.cuda.memory_allocated() / 2**20,  # bytes to mbs
            "gpu_reserved_mb": torch.cuda.memory_reserved() / 2**20,
        }
    return {"gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0}
