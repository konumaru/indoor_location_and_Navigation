import os
import glob
import joblib
import pathlib
import numpy as np
import pandas as pd
from rich.progress import track

from typing import Dict

from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache
from utils.feature import FeatureStore


def main():
    src_dir = pathlib.Path("../data/raw/train/")
    filepaths = [
        path_filepath
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    ]

    for filepath in track(filepaths):
        site_id = filepath.parent.parent.name
        floor = filepath.parent.name
        path_id = filepath.name.split(".")[0]

        feature = FeatureStore(
            site_id=site_id, floor=floor, path_id=path_id, input_path="../data/raw/"
        )
        feature.load_all_data()
        feature.save()


if __name__ == "__main__":
    with timer("ParseData"):
        main()
