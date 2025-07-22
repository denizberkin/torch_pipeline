import os
import re
import inspect
import importlib
from typing import Optional


from utils.logger import get_logger


def find_class_by_alias(alias: str, dir: str) -> Optional[object]:
    """
    Iterates all classes in given directory and returns if name == str(class)
    ### Parameters
        - alias: alias to search within the `get_alias` method of classes
        - dir: directory to search for modules
    ### Returns
        - found: class object if found, otherwise None
    ### Signature
        (str, str) -> Union[object, None]
    """
    logger = get_logger()
    found = None
    files = {fn: cur_dir for cur_dir, _, file_list in os.walk(dir) for fn in file_list}
    
    for fn, cd in files.items():
        if fn.startswith("__") or fn in {"base.py", "build.py"} or not fn.endswith(".py"):
            continue  # pass unnecessary files

        f, _ = os.path.splitext(fn)
        module_name = f"""{'.'.join(re.split(r"[\\/]+", cd))}.{f}"""  # check sub-modules iteratively
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Error importing module {module_name}: {e}")
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if hasattr(cls, "get_alias") and "Base" not in str(cls):
                if cls.get_alias(cls) == alias:
                    found = cls
                if cls.get_alias(cls) == "base":  # throw another warning if they miss
                    logger.warning(f"Class '{cls}' does not define 'get_alias' which is a must!")
    if not found:
        logger.warning(f"Class with alias '{alias}' not found in directory '{dir}'")
    return found
