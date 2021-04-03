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
from utils import load_pickle, dump_pickle, save_cache


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


# === waypint, (site, floor, path. timestamp, x, y), string ===


@save_cache("../data/working/train_waypint.pkl", use_cache=True)
def create_train_waypoint():
    src_dir = pathlib.Path("../data/raw/train/")
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_data_from_pathtxt)(path_filepath, "TYPE_WAYPOINT", True)
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    )
    waypoint = np.concatenate(data, axis=0)
    return waypoint


@save_cache("../data/working/test_waypint.pkl", use_cache=True)
def create_test_waypoint():
    waypoint = pd.read_csv("../data/raw/sample_submission.csv")
    waypoint[["site", "path", "timestamp"]] = waypoint["site_path_timestamp"].str.split(
        "_", expand=True
    )
    waypoint.drop(["site_path_timestamp"], axis=1, inplace=True)

    waypoint = waypoint[["site", "floor", "path", "timestamp", "x", "y"]].astype(str)
    waypoint = waypoint.to_numpy()
    return waypoint


# === wifi, (bssid, rssi, frequency, ts_diff). ===


@save_cache("../data/working/train_wifi.pkl", False)
def create_train_wifi():
    def get_wifi_from_waypoints(waypoint, max_len=100):
        (site, floor, path, timestamp, x, y) = waypoint
        path_filepath = pathlib.Path(f"../data/raw/train/{site}/{floor}/{path}.txt")
        wifi = get_data_from_pathtxt(path_filepath, "TYPE_WIFI")

        extract_idx = [2, 3, 4]
        data = np.concatenate(
            [
                np.tile("nan", (1, 100)).astype("<U40"),  # bssid
                np.tile("-999", (1, 100)).astype("<U40"),  # rssi
                np.tile("0", (1, 100)).astype("<U40"),  # frequency
                np.tile("-999", (1, 100)).astype("<U40"),  # ts_diff
            ],
            axis=0,
        )

        if len(wifi) > 0:
            ts_diff = wifi[:, 0].astype("int64") - timestamp.astype("int64")
            ts_diff_min = np.abs(np.min(ts_diff))
            # Extract latest values, except feature information.
            wifi = wifi[(ts_diff <= ts_diff_min)]
            # Select feature diff of timestamp.
            ts_diff = ts_diff[(ts_diff <= ts_diff_min)]
            # Extract columns of (bssid, rssi, frequency).
            wifi = wifi[:, extract_idx]
            # Sort values by rssi.
            sort_idx = np.argsort(wifi[:, 1])
            wifi = wifi[sort_idx]
            ts_diff = ts_diff[sort_idx]

            data[:3, : wifi.T.shape[1]] = wifi.T[:, :max_len]
            data[3, : wifi.T.shape[1]] = ts_diff[:max_len]
        return data

    waypoints = load_pickle("../data/working/train_waypint.pkl")
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_wifi_from_waypoints)(waypoint)
        for waypoint in track(waypoints)
    )
    data = np.array(data)
    return data


@save_cache("../data/working/test_wifi.pkl", False)
def create_test_wifi():
    def get_test_wifi_from_waypoints(
        waypoint: np.ndarray, max_len: int = 100
    ) -> np.ndarray:
        (site, floor, path, timestamp, x, y) = waypoint
        path_filepath = pathlib.Path(f"../data/raw/test/{path}.txt")
        wifi = get_data_from_pathtxt(path_filepath, "TYPE_WIFI")

        extract_idx = [2, 3, 4]
        data = np.concatenate(
            [
                np.tile("nan", (1, 100)).astype("<U40"),  # bssid
                np.tile("-999", (1, 100)).astype("<U40"),  # rssi
                np.tile("0", (1, 100)).astype("<U40"),  # frequency
                np.tile("-999", (1, 100)).astype("<U40"),  # ts_diff
            ],
            axis=0,
        )

        if len(wifi) > 0:
            ts_diff = wifi[:, 0].astype("int64") - int(timestamp)
            ts_diff_min = np.abs(np.min(ts_diff))
            # Add feature diff of timestamp.
            wifi = np.concatenate([wifi, ts_diff.reshape(-1, 1)], axis=1)
            # Extract latest values, except feature information.
            wifi = wifi[(ts_diff <= ts_diff_min)]
            # Select feature diff of timestamp.
            ts_diff = ts_diff[(ts_diff <= ts_diff_min)]
            # Extract columns of (bssid, rssi, frequency).
            wifi = wifi[:, extract_idx]
            # Sort values by rssi.
            sort_idx = np.argsort(wifi[:, 1])
            wifi = wifi[sort_idx]
            ts_diff = ts_diff[sort_idx]

            data[:3, : wifi.T.shape[1]] = wifi.T[:, :max_len]
            data[3, : wifi.T.shape[1]] = ts_diff[:max_len]
        return data

    waypoints = load_pickle("../data/working/test_waypint.pkl")
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_test_wifi_from_waypoints)(waypoint)
        for waypoint in track(waypoints)
    )
    data = np.array(data)
    return data


def main():
    # Create data
    print("Create waypoint ...")
    _ = create_train_waypoint()
    _ = create_test_waypoint()

    print("Create wifi ... ")
    _ = create_train_wifi()
    _ = create_test_wifi()


if __name__ == "__main__":
    with timer("ParseData"):
        main()
