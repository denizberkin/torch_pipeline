from typing import List, Union

import omegaconf as oc

from utils.schema import ConfigSchema


def load_config(file_path: Union[str, List[str]], strict: bool = False) -> oc.DictConfig:
    """
    Load a configuration file and return it as an omegaconf DictConfig object.
    :param file_path (str or List[str]): Path to the configuration file(s).
    :param strict (bool): If True, will enforce strict checking on the config.
    """
    if isinstance(file_path, str):
        file_path = [file_path]
    config = oc.OmegaConf.create()
    for path in file_path:
        loaded_config = oc.OmegaConf.load(path)
        config = oc.OmegaConf.merge(config, loaded_config)
    if strict:
        return add_strict_checking(config)
    return config


def add_strict_checking(config: oc.DictConfig) -> oc.DictConfig:
    """
    To add strict checks on config integrity with the help of schema.py
    Throws an error if config does not match the schema.
    """
    schema = oc.OmegaConf.structured(ConfigSchema)
    return oc.OmegaConf.merge(schema, config)
