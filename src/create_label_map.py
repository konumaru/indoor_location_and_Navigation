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


@save_cache("../data/label_encode/map_site_id.pkl", False)
def create_site_map(filepaths: List):
    def get_site_id_from_feature_store(filepath):
        feature = load_pickle(filepath, verbose=False)
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
        feature = load_pickle(filepath, verbose=False)
        uniques = feature.wifi.bssid.unique()
        if len(uniques) > 0:
            return uniques
        else:
            return np.array([])

    bssid = Parallel(n_jobs=-1)(
        delayed(get_bssid_from_featureStore)(filepath) for filepath in track(filepaths)
    )

    bssid = np.concatenate(bssid, axis=0)
    unique_bssid = np.unique(bssid)

    bssid_map = {_bssid: i + 1 for i, _bssid in enumerate(unique_bssid)}
    return bssid_map


@save_cache("../data/label_encode/map_beacon_uuid.pkl", False)
def create_beacon_uuid_map(filepaths: List):
    def get_beacon_uuid_from_feature_store(filepath):
        feature = load_pickle(filepath, verbose=False)
        uniques = feature.beacon.uuid.unique()
        if len(uniques) > 0:
            return uniques
        else:
            return np.array([])

    uuid = Parallel(n_jobs=-1)(
        delayed(get_beacon_uuid_from_feature_store)(filepath)
        for filepath in track(filepaths)
    )

    uuid = np.concatenate(uuid, axis=0)
    unique_uuid = np.unique(uuid)

    uuid_map = {_uuid: i + 1 for i, _uuid in enumerate(unique_uuid)}
    return uuid_map


def main():
    # Train files.
    src_dir = pathlib.Path("../data/working/")
    filepaths = [
        path_filepath
        for path_filepath in src_dir.glob("*")
        if path_filepath.name != ".gitkeep"
    ]
    # Test files
    src_test_dir = pathlib.Path("../data/submit/path_data/")
    filepaths += [
        path_filepath
        for path_filepath in src_test_dir.glob("*")
        if path_filepath.name != ".gitkeep"
    ]

    _ = create_site_map(filepaths)
    _ = create_bssid_map(filepaths)
    _ = create_beacon_uuid_map(filepaths)


if __name__ == "__main__":
    with timer("Create label map"):
        main()
