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
    task: str
    root: str
    batch_size: int
    val_split: Optional[float]
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class OptimConfig:
    name: str
    lr: float
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    pretrained_path: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentTrackerConfig:
    enabled: bool
    name: str
    experiment_name: str
    port: int
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class TrainConfig:
    epochs: int
    log_interval: int
    save_interval: int


@dataclass
class ConfigSchema:
    seed: int
    device: str
    logging: LoggingConfig
    losses: List[LossConfig]
    metrics: List[MetricConfig]
    data: DatasetConfig
    optim: OptimConfig
    model: ModelConfig
    tracker: ExperimentTrackerConfig
    train: TrainConfig
