import os

from losses import *
from losses.build import build_loss
from utils.config import load_config


def main():
    conf_folder = os.path.join(os.path.dirname(__file__), "configs")
    config_filenames = [os.path.join(conf_folder, fn) for fn in os.listdir(conf_folder)]
    config = load_config(config_filenames, strict=True)
    for lossfn in config.losses:
        print(f"Loss function: {lossfn.name}, weight: {lossfn.weight}")


if __name__ == "__main__":
    main()
