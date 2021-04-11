import os
import glob
import pathlib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from rich.progress import track

from typing import Dict


from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache
from utils.feature import FeatureStore


@save_cache("../data/label_encode/map_bssid.pkl")
def create_bssid_map():
    def get_bssid_from_featureStore(filepath):
        site_id = filepath.parent.parent.name
        floor = filepath.parent.name
        path_id = filepath.name.split(".")[0]

        feature = load_pickle(f"../data/working/{path_id}.pkl", verbose=False)
        uniques = feature.wifi.bssid.unique()
        if len(uniques) > 0:
            return uniques
        else:
            return np.array([])

    src_dir = pathlib.Path("../data/raw/train/")
    filepaths = [
        path_filepath
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    ]

    bssid = Parallel(n_jobs=-1)(
        delayed(get_bssid_from_featureStore)(filepath) for filepath in track(filepaths)
    )
    bssid = np.concatenate(bssid, axis=0)
    unique_bsid = np.unique(bssid)

    bssid_map = {_bssid: i for i, _bssid in enumerate(bssid)}
    return bssid_map


def main():
    _ = create_bssid_map()


if __name__ == "__main__":
    with timer("Create label map"):
        main()
