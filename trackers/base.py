from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """
    TODO: only abstract available, add mlflow and similar trackers
    """
    @abstractmethod
    def start(self): raise NotImplementedError
    @abstractmethod
    def log_params(self): raise NotImplementedError
    @abstractmethod
    def log_metrics(self): raise NotImplementedError
    @abstractmethod
    def log_artifacts(self): raise NotImplementedError
    @abstractmethod
    def end(self): raise NotImplementedError
    def get_alias(self): return "base"
