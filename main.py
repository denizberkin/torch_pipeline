import os

from losses import *
from utils.config import load_config

# from losses.build import build_loss
from utils.utils import find_class_by_name


def main():
    conf_folder = os.path.join(os.path.dirname(__file__), "configs")
    config_filenames = [os.path.join(conf_folder, fn) for fn in os.listdir(conf_folder)]
    config = load_config(config_filenames, strict=True)


if __name__ == "__main__":
    main()
