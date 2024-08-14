from absl import logging
import numpy as np


def compare_values(v1, v2, path=""):
    if isinstance(v1, dict) and isinstance(v2, dict):
        return compare_dicts(v1, v2, path)
    elif isinstance(v1, list) and isinstance(v2, list):
        return compare_lists(v1, v2, path)
    elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        if not np.array_equal(v1, v2):
            logging.error(f"Difference found at {path} (numpy arrays differ)")
            return False
        return True
    else:
        if v1 != v2:
            logging.error(f"Difference found at {path}: {v1} != {v2}")
            return False
        return True


def compare_dicts(d1, d2, path=""):
    all_equal = True
    for key in d1.keys():
        new_path = f"{path}/{key}"
        if key not in d2:
            logging.error(f"Key {key} found in first dictionary but not in second at path {new_path}")
            all_equal = False
        else:
            if not compare_values(d1[key], d2[key], new_path):
                all_equal = False

    for key in d2.keys():
        if key not in d1:
            new_path = f"{path}/{key}"
            logging.error(f"Key {key} found in second dictionary but not in first at path {new_path}")
            all_equal = False
    return all_equal


def compare_lists(l1, l2, path=""):
    if len(l1) != len(l2):
        logging.error(f"Different lengths at {path}: {len(l1)} != {len(l2)}")
        return False

    all_equal = True
    for i, (item1, item2) in enumerate(zip(l1, l2)):
        new_path = f"{path}[{i}]"
        if not compare_values(item1, item2, new_path):
            all_equal = False
    return all_equal


def compare_structures(var1, var2):
    logging.info("Comparing of data")
    if isinstance(var1, list) and isinstance(var2, list):
        compare_lists(var1, var2)
    else:
        logging.error("The provided variables are not lists of dictionaries.")


