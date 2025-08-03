
import os
import argparse
from utils.config import ConfigSchema, load_config


def args_parser():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="path to yaml within configs/ folder")
    args = parser.parse_args()
    return args  


def config_parser(config_path: str, strict: bool = True) -> ConfigSchema:
    """
    Interface to check for proper config path and load the config accordingly.
    ### Args
        - config_path (str): Path to the configuration file.
        - strict (bool): 
            If True, strictly compares the config file with /utils/schema.py,
            raises an error if there are mismatches. 
            If False, loads the config without type and name checks.
    ### Returns
        - ConfigSchema object with the loaded configuration.
    ### Signature
        - (str) -> ConfigSchema
    """
    try:  # either provide path or filename within configs folder 
       config = load_config(config_path, strict=strict)
    except FileNotFoundError as e:
        conf_folder = os.path.join(os.path.dirname(__file__), "configs")
        config_fn = os.path.join(conf_folder, config_path)
        config = load_config(config_fn, strict=strict)
    return config
