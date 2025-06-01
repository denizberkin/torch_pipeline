from torch.utils.data import DataLoader

from data.base import BaseDataset
from utils.logger import get_logger
from utils.schema import DatasetConfig
from utils.utils import find_class_by_alias


def build_data_loaders(config: DatasetConfig, _type: str) -> DataLoader:
    """
    Build the dataloader based on the provided configuration.
    ### Args:
        config (DatasetConfig): Configuration dictionary for the dataloader.
        _type (str): Type of dataset to which DataLoader instance will be initialized enum["train", "val", "test"]
    ### Returns:
        DataLoader: Configured dataloader instance.
    """
    logger = get_logger()
    dataset: BaseDataset = find_class_by_alias(config.name, "data")(config.root, _type, **config.kwargs)
    logger.info(
        f"Using {config.name} {_type} set from current module: {dataset.__class__.__name__}, module name: {__name__}"
    )
    shuffle = True if _type == "train" else False
    return DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle)
