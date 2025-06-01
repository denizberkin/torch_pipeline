import torch
import torch.nn as nn

from models.base import BaseModel
from utils.logger import get_logger
from utils.schema import ModelConfig
from utils.utils import find_class_by_alias


def build_models(config: ModelConfig, device: torch.device = None) -> nn.Module:
    """
    Build models based on the provided configuration.
    ### Parameters
        - config: Configuration dictionary containing model settings.
    ### Returns
        - An instance of the model.
    """
    logger = get_logger()
    model: BaseModel = find_class_by_alias(config.name, "models")(**config.kwargs)
    if config.pretrained:
        try:
            model.load_state_dict(torch.load(config.pretrained_path, map_location=device))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained model from {config.pretrained}. \
                               Check path or save file, it should be a state dict!"
            )
    logger.info(f"Model {model.get_alias()} built with {model.get_num_params()} parameters.")
    logger.info(f"Num params:\t\t {model.get_num_params()}")
    logger.info(f"Num trainable params:\t {model.get_trainable_params()}")
    return model
