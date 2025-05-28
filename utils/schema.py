from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LoggingConfig:
    enabled: bool
    level: str
    log_dir: Optional[str] = None  # default: ./output/logs
    log_file: Optional[str] = None  # default: {experiment_name}.log


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
    logging: LoggingConfig
    losses: List[LossConfig]
    metrics: List[MetricConfig]
    dataset: List[DatasetConfig]
