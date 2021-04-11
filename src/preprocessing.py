import pathlib
import numpy as np
import pandas as pd
from rich.progress import track
from joblib import Parallel, delayed

from typing import List

from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache
from utils.feature import FeatureStore


@save_cache("../data/preprocessing/train_waypoint.pkl", True)
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
    waypoint = waypoint.sort_values(by=["path", "timestamp"]).reset_index(drop=True)
    return waypoint


@save_cache("../data/preprocessing/train_wifi_results.pkl", True)
def create_wifi():
    def get_wifi_feature(path_id, gdf):
        seq_len = 100
        bssid = []
        rssi = []
        freq = []

        feature = load_pickle(f"../data/working/{path_id}.pkl", verbose=False)
        wifi = feature.wifi.copy()
        wifi["bssid"] = wifi["bssid"].map(bssid_map)

        min_idx = gdf.index.min()
        max_idx = gdf.index.max()

        for i, row in gdf.iterrows():
            ts_pre_wp = gdf.loc[i - 1, "timestamp"] if i > min_idx else None
            ts_current_wp = gdf.loc[i, "timestamp"]
            ts_post_wp = gdf.loc[i + 1, "timestamp"] if (i + 1) < max_idx else None

            _wifi = wifi.copy()
            # NOTE: ターゲットとなるwaypointとその前後のwaypointの間にあるデータを取得する。
            ts_wifi = _wifi["timestamp"].values
            pre_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_pre_wp == None
                else (ts_pre_wp < ts_wifi)
            )
            psot_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_post_wp == None
                else (ts_wifi < ts_post_wp)
            )
            _wifi = _wifi[pre_flag & psot_flag]

            _wifi = _wifi.sort_values(by="rssi", ascending=False)
            _wifi = _wifi.head(seq_len)

            _bssid = np.zeros(seq_len)
            _rssi = np.tile(-999, seq_len)
            _freq = np.tile(-999, seq_len)

            _bssid[: len(_wifi)] = _wifi["bssid"].astype("int32").to_numpy()
            _rssi[: len(_wifi)] = _wifi["rssi"].astype("float32").to_numpy()
            _freq[: len(_wifi)] = _wifi["frequency"].astype("float32").to_numpy()

            bssid.append(_bssid)
            rssi.append(_rssi)
            freq.append(_freq)

        return bssid, rssi, freq

    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    bssid_map = load_pickle("../data/label_encode/map_bssid.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_wifi_feature)(path_id, gdf)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


def create_wifi_features():
    results = create_wifi()
    bssid, rssi, freq = zip(*results)

    bssid = np.concatenate(bssid, axis=0)
    np.save("../data/preprocessing/train_wifi_bssid.npy", bssid)

    rssi = np.concatenate(rssi, axis=0)
    np.save("../data/preprocessing/train_wifi_rssi.npy", rssi)

    freq = np.concatenate(freq, axis=0)
    np.save("../data/preprocessing/train_wifi_freq.npy", freq)


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

    print("Create wifi ...")
    _ = create_wifi_features()


if __name__ == "__main__":
    with timer("Creat Features"):
        main()
