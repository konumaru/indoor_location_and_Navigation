import os
import time
import pickle
import datetime
from contextlib import contextmanager
from typing import List, Any


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    druration_time = str(datetime.timedelta(seconds=time.time() - start_time))[:7]
    print(f"[{name}] done in {druration_time}\n")


def load_pickle(filepath: str, verbose: bool = True):
    if verbose:
        print(f"Load pickle from {filepath}")
    with open(filepath, "rb") as file:
        return pickle.load(file)


def dump_pickle(filepath: str, data: Any, verbose: bool = True):
    if verbose:
        print(f"Dump pickle to {filepath}")
    with open(filepath, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def use_cache(filepath: str, use_cache: bool = True):
    """Save return dataframe with pickle.
    Parameters
    ----------
    filename : str
        filename, when save with pickle.
    use_cache : bool, optional
        Is use already cash result then pass method process, by default True
    Example
    -------
    from mikasa.io import save_cache
    @save_cache("path/to/file.pkl", use_cache=False)
    def create_feature(data):
        feature_name = "..."
        return data[target_name]
    """

    def wrapper(func):
        def run_func(*args, **kwargs):
            if os.path.exists(filepath) and use_cache:
                print(f"Load cache from {filepath}")
                with open(filepath, "rb") as file:
                    return pickle.load(file)

            result = func(*args, **kwargs)
            print(f"Dump cache to {filepath}")
            with open(filepath, "wb") as file:
                pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            return result

        return run_func

    return wrapper
