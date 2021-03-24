import os
import glob
import joblib
import pathlib
import numpy as np
import pandas as pd
from rich.progress import track

from typing import Dict

import config

from utils import timer

# ==== waypoint ===


def get_data_from_pathtxt(
    filepath: pathlib.PosixPath, data_type: str, is_join_ids: bool = False
) -> np.ndarray:
    with open(filepath) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        tmp = line.strip().split("\t")
        if tmp[1] == data_type:
            data.append(tmp)

    data = np.array(data)
    # Drop data_type column.
    if data.shape[0] > 0:
        data = np.delete(data, 1, axis=1)
    # Concatenate site, floor and path.
    if is_join_ids:
        site_id = filepath.parent.parent.name
        floor_id = filepath.parent.name
        path_id = filepath.name.split(".")[0]
        site_floor_path = np.tile([site_id, floor_id, path_id], (data.shape[0], 1))
        data = np.concatenate([site_floor_path, data], axis=1)
    return data


def get_waypoint_in_parallel(
    src_dir: str, data_type: str, is_join_ids: bool = False
) -> np.ndarray:
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_data_from_pathtxt)(path_filepath, data_type, is_join_ids)
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    )
    data = np.concatenate(data, axis=0)
    return data


# === Wifi ===


def get_wifi_from_waypoints(waypoint, max_len=100):
    (site, floor, path, timestamp, x, y) = waypoint
    path_filepath = pathlib.Path(f"../data/raw/train/{site}/{floor}/{path}.txt")
    wifi = get_data_from_pathtxt(path_filepath, "TYPE_WIFI")

    extract_idx = [2, 3, 4]
    data = np.full((len(extract_idx), max_len), np.nan, dtype="<U40")

    if len(wifi) > 0:
        ts_diff = wifi[:, 0].astype("int64") - timestamp.astype("int64")
        ts_diff_min = np.abs(np.min(ts_diff))
        # Extract latest values, except feature information.
        wifi = wifi[(ts_diff <= ts_diff_min)]
        # Extract columns of (bssid, rssi, frequency).
        wifi = wifi[:, extract_idx]
        # Sort values by rssi.
        sort_idx = np.argsort(wifi[:, 1])
        wifi = wifi[sort_idx]

        data = np.full((len(extract_idx), max_len), np.nan, dtype="<U40")
        data[:, : wifi.T.shape[1]] = wifi.T[:, :max_len]
    return data


def get_wifi_from_waypoints_in_parallel(waypoints: np.ndarray) -> np.ndarray:
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_wifi_from_waypoints)(waypoint)
        for waypoint in track(waypoints)
    )
    data = np.array(data)
    return data


def main():
    src_dir = pathlib.Path("../data/raw/train/")
    # Data columns is (site, floor, path. timestamp, x, y).
    with timer("Get waypoint of train"):
        waypoints = get_waypoint_in_parallel(src_dir, "TYPE_WAYPOINT", is_join_ids=True)
    np.save("../data/working/train_waypoint.npy", waypoints)

    with timer("Get wifi of train"):
        wifi_features = get_wifi_from_waypoints_in_parallel(waypoints)
    np.save("../data/working/train_wifi_features.npy", wifi_features)


if __name__ == "__main__":
    with timer("ParseData"):
        main()
