import os

from losses import *
from losses.build import build_loss
from utils.config import check_config_integrity, load_config


def main():
    conf_folder = os.path.join(os.path.dirname(__file__), "configs")
    config_filenames = [os.path.join(conf_folder, fn) for fn in os.listdir(conf_folder)]
    config = load_config(config_filenames)
    config = check_config_integrity(config)


if __name__ == "__main__":
    main()
