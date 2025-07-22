from typing import Dict, List

from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, NormalizedRootMeanSquaredError
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU
from torchmetrics.text import EditDistance

from metrics.sklearn_wrapper import Accuracy, F1Score, Precision, Recall
from utils.constants import METRICS_DIR
from utils.logger import get_logger
from utils.utils import find_class_by_alias
from utils.schema import MetricConfig


def build_metrics(cfg_list: List[MetricConfig]) -> Dict[str, object]:
    """ Build the metrics module. """
    logger = get_logger()
    metrics = {}
    for cfg in cfg_list:
        alias = cfg.name
        if alias in PREDEFINED_METRICS:
            cls = PREDEFINED_METRICS[alias]
        else:
            cls = find_class_by_alias(alias, METRICS_DIR)

        if cls is None:
            continue
        logger.info(f"Using class '{alias}', module name: {__name__}")
        kwargs = dict(cfg.kwargs) if getattr(cfg, "kwargs", None) else {}
        metric_fn = cls(**kwargs)
        metrics[alias] = {"metric_fn": metric_fn}  # is dict so can be extended if needed
    return metrics


PREDEFINED_METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1": F1Score,
    "map": MeanAveragePrecision,
    "iou": IntersectionOverUnion,
    "mean_iou": MeanIoU,
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mape": MeanAbsolutePercentageError,
    "nrmse": NormalizedRootMeanSquaredError,
    "edit_distance": EditDistance,
}
