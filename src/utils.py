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
