import os
import time
import datetime
from contextlib import contextmanager


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    druration_time = str(datetime.timedelta(seconds=time.time() - start_time))[:7]
    print(f"[{name}] done in {druration_time}\n")
