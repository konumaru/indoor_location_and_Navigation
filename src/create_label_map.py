import os
import glob
import pathlib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from rich.progress import track

from typing import List


from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache
from utils.feature import FeatureStore


@save_cache("../data/label_encode/map_site_id.pkl", True)
def create_site_map(filepaths: List):
    def get_site_id_from_feature_store(filepath):
        path_id = filepath.name.split(".")[0]

        feature = load_pickle(f"../data/working/{path_id}.pkl", verbose=False)
        return feature.site_id

    site_ids = Parallel(n_jobs=-1)(
        delayed(get_site_id_from_feature_store)(filepath)
        for filepath in track(filepaths)
    )
    unique_site_ids = np.unique(site_ids)
    siteId_map = {site_id: i + 1 for i, site_id in enumerate(unique_site_ids)}
    return siteId_map


@save_cache("../data/label_encode/map_bssid.pkl", False)
def create_bssid_map(filepaths: List):
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

    bssid = Parallel(n_jobs=-1)(
        delayed(get_bssid_from_featureStore)(filepath) for filepath in track(filepaths)
    )
    bssid = np.concatenate(bssid, axis=0)
    unique_bsid = np.unique(bssid)

    bssid_map = {_bssid: i + 1 for i, _bssid in enumerate(unique_bsid)}
    return bssid_map


def main():
    src_dir = pathlib.Path("../data/raw/train/")
    filepaths = [
        path_filepath
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    ]

    _ = create_site_map(filepaths)
    _ = create_bssid_map(filepaths)


if __name__ == "__main__":
    with timer("Create label map"):
        main()
