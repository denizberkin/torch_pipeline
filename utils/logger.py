import logging
import os
import sys
import time
from typing import Callable

import coloredlogs

LOGGER: logging.Logger = None


def get_logger(path: str = None, base_level: int = logging.INFO) -> logging.Logger:
    # Used a global logger to prevent multiple instances of
    # loggers running at the same time.
    _format = "%(asctime)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(_format)

    global LOGGER
    if LOGGER is not None:
        if LOGGER.level != base_level:
            LOGGER.setLevel(base_level)
        if path:
            # remove other file handlers
            if LOGGER.hasHandlers():
                LOGGER.removeHandler(next(filter(lambda x: isinstance(x, logging.FileHandler), LOGGER.handlers), None))
            LOGGER.addHandler(get_file_handler(path, formatter=formatter, base_level=base_level))
        return LOGGER

    logger = logging.getLogger("LoggerSingleton")
    logger.propagate = False
    logger.setLevel(base_level)

    # stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(base_level)
    stdout_handler.setFormatter(formatter)

    # log to file if path != None
    if path:
        file_handler = get_file_handler(path, formatter, base_level=base_level)

        logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    LOGGER = logger
    try:
        coloredlogs.install(level=base_level, logger=LOGGER, fmt=_format)
    except NameError:
        print("coloredlogs either not defined or is not installed, defaulting to normal logging.")
    return logger


def get_file_handler(
    path: str,
    formatter: logging.Formatter,
    base_level: int = logging.INFO,
) -> logging.FileHandler:
    folder = os.path.dirname(path)
    fn = os.path.basename(path)
    if not os.path.isdir(folder) and bool(folder):
        os.makedirs(folder)
    if fn.split(".")[-1] != "log":
        print("Please specify extension of logger path as .log")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(base_level)
    file_handler.setFormatter(formatter)

    return file_handler


def log_timeit(func: Callable) -> Callable:
    def timeit_wrapper(*args, **kwargs):
        try:
            logger = kwargs["logger"]
        except KeyError:
            logger = get_logger()  # logger is Singleton, returns the original
        start_time = time.perf_counter()
        func_name = str(func.__name__).split("(")[0]
        logger.critical("Function ---%s--- Started!", func_name)
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_time, time_scale = (total_time, "seconds") if total_time > 0.1 else (total_time * 1000, "milliseconds")

        logger.critical("Function ---%s--- Done! - Took %s %s.", func_name, total_time, time_scale)
        return result

    return timeit_wrapper
