from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LossConfig:
    name: str
    weight: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class MetricConfig:
    name: str
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class DatasetConfig:
    name: str
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ConfigSchema:
    seed: int
    device: str
    losses: List[LossConfig]
    metrics: List[MetricConfig]
    dataset: List[DatasetConfig]
