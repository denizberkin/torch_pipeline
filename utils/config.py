from dataclasses import dataclass, fields, is_dataclass
from typing import List, Union, get_args, get_origin, get_type_hints

import omegaconf as oc

from utils.logger import get_logger
from utils.schema import ConfigSchema


def load_config(file_path: Union[str, List[str]], strict: bool = False) -> Union[oc.DictConfig, ConfigSchema]:
    """
    Load a configuration file and return it as an omegaconf DictConfig object.
    ### Parameters
        - file_path: Path to the configuration file(s).
        - strict: whether to enable strict checking of config, raise `KeyError` if issues found.
    ### Returns
        - omegaconf.DictConfig object which has the same structure as ConfigSchema.
    ### Signature
        - (union[str, list[str]], bool) -> DictConfig | ConfigSchema
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
    merged = oc.OmegaConf.merge(schema, config)
    validate_keys(ConfigSchema, merged)
    return merged


def validate_keys(schema: type, merged: oc.DictConfig, path: str = "") -> None:
    """
    Recursively validate keys in the merged configuration against the schema.
    ### Parameters
        - schema: The schema to validate against, a dataclass or ConfigSchema.
        - merged: The merged configuration to validate.
    ### Returns
        - None, raises KeyError if any required key is missing.
    ### Signature
        - (type, DictConfig, str) -> None
    """
    logger = get_logger()
    for field in fields(schema):
        key = field.name
        sub_path = f"{path}.{key}" if path else key
        type_hint = get_type_hints(schema).get(key)  # mostly == obj.__annotations__
        is_optional = get_origin(type_hint) is Union and type(None) in get_args(type_hint)  # check if field is optional
        if key not in merged:
            print(f"Key '{key}' not found in merged config at path '{sub_path}'")
            if not is_optional:
                logger.error(f"KeyError: Missing required key '{sub_path}' in config.")
                raise KeyError(f"Missing required key '{sub_path}' in config.")
            continue  # skip optional keys, no need to check further

        val = merged[key]
        if is_dataclass(type_hint) and isinstance(val, oc.DictConfig):
            validate_keys(type_hint, val, sub_path)
        elif is_list_of_dataclass(type_hint):
            item_type = get_args(type_hint)[0]
            for i, item in enumerate(val):
                if isinstance(item, oc.DictConfig):
                    validate_keys(item_type, item, f"{sub_path}[{i}]")


def is_list_of_dataclass(type_hint):
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    return origin == list and len(args) == 1 and is_dataclass(args[0])
