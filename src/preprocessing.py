import pathlib
import numpy as np
import pandas as pd
from rich.progress import track
from joblib import Parallel, delayed

from typing import List

from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache
from utils.feature import FeatureStore


@save_cache("../data/preprocessing/train_waypoint.pkl")
def create_waypoint(filepaths: List):
    def get_waypoint_from_featureStore(filepath):
        path_id = filepath.name.split(".")[0]

        feature = load_pickle(f"../data/working/{path_id}.pkl", verbose=False)
        wp = feature["waypoint"]
        wp["site"] = feature.site_id
        wp["floor"] = feature.n_floor
        wp["path"] = feature.path_id
        if len(wp) > 0:
            return wp
        else:
            return pd.DataFrame([])

    waypoint = Parallel(n_jobs=-1)(
        delayed(get_waypoint_from_featureStore)(filepath)
        for filepath in track(filepaths)
    )
    waypoint = pd.concat(waypoint, axis=0).reset_index(drop=True)
    return waypoint


def main():
    src_dir = pathlib.Path("../data/raw/train/")
    filepaths = [
        path_filepath
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    ]

    print("Create waypoint ...")
    _ = create_waypoint(filepaths)


if __name__ == "__main__":
    with timer("Creat Features"):
        main()
