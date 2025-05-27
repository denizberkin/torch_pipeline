import importlib
import inspect
import os
from typing import Optional

from utils.logger import get_logger


def find_class_by_name(name: str, dir: str) -> Optional[object]:
    """
    Iterates all classes in given directory and returns if name == str(class)
    :param name aaaa
    """
    logger = get_logger()
    found = None
    for fn in os.listdir(dir):
        if fn.startswith("__") or fn in {"base.py", "build.py"} or not fn.endswith(".py"):
            continue  # pass other files

        f, ext = os.path.splitext(fn)
        module_name = f"{dir}.{f}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Error importing module {module_name}: {e}")
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if hasattr(cls, "get_alias"):
                if cls.get_alias(cls) == name:
                    found = cls

    if not found:
        logger.warning(f"Class '{name}' not found in directory '{dir}'")
    return found
