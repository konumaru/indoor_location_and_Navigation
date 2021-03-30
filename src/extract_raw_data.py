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
from utils import load_pickle, dump_pickle


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


# ==== waypoint ===


def get_waypoint_in_parallel(
    src_dir: str, data_type: str, is_join_ids: bool = False
) -> np.ndarray:
    src_dir = pathlib.Path(src_dir)
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_data_from_pathtxt)(path_filepath, data_type, is_join_ids)
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    )
    # Data columns is (site, floor, path. timestamp, x, y).
    data = np.concatenate(data, axis=0)
    return data


def get_test_waypoint():
    waypoint = pd.read_csv("../data/raw/sample_submission.csv")
    waypoint[["site", "path", "timestamp"]] = waypoint["site_path_timestamp"].str.split(
        "_", expand=True
    )
    waypoint.drop(["site_path_timestamp"], axis=1, inplace=True)

    waypoint = waypoint[["site", "floor", "path", "timestamp", "x", "y"]].astype(str)
    waypoint = waypoint.to_numpy()
    return waypoint


def waypoint_preprocessing():
    # Train data
    filepath = "../data/working/train_waypoint.npy"
    if not pathlib.Path(filepath).exists():
        with timer("Dump waypoint of train"):
            waypoints = get_waypoint_in_parallel(
                "../data/raw/train/", "TYPE_WAYPOINT", True
            )
            np.save(filepath, waypoints)
    # Test data
    filepath = "../data/working/test_waypoint.npy"
    if not pathlib.Path(filepath).exists():
        with timer("Dump waypoint of test"):
            waypoints = get_test_waypoint()
            np.save(filepath, waypoints)


# === Wifi ===


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
        ],
        axis=0,
    )

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

        data[:, : wifi.T.shape[1]] = wifi.T[:, :max_len]
    return data


def get_wifi_from_waypoints_in_parallel() -> np.ndarray:
    waypoints = np.load("../data/working/train_waypoint.npy")
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_wifi_from_waypoints)(waypoint)
        for waypoint in track(waypoints)
    )
    data = np.array(data)
    return data


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
        ],
        axis=0,
    )

    if len(wifi) > 0:
        ts_diff = wifi[:, 0].astype("int64") - int(timestamp)
        ts_diff_min = np.abs(np.min(ts_diff))
        # Extract latest values, except feature information.
        wifi = wifi[(ts_diff <= ts_diff_min)]
        # Extract columns of (bssid, rssi, frequency).
        wifi = wifi[:, extract_idx]
        # Sort values by rssi.
        sort_idx = np.argsort(wifi[:, 1])
        wifi = wifi[sort_idx]

        data[:, : wifi.T.shape[1]] = wifi.T[:, :max_len]
    return data


def get_test_wifi_from_waypoints_in_parallel() -> None:
    waypoints = np.load("../data/working/test_waypoint.npy", allow_pickle=True)
    data = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_test_wifi_from_waypoints)(waypoint)
        for waypoint in track(waypoints)
    )
    data = np.array(data)
    return data


def dump_wifi_bssid_map():
    wifi_train = np.load("../data/working/train_wifi_features.npy")
    wifi_test = np.load("../data/working/test_wifi_features.npy")
    # wifi columns is (bssid, rssi, frequency).
    bssid_uniques = np.unique(
        np.concatenate([wifi_train[:, 0], wifi_test[:, 0]], axis=0).ravel()
    )
    bssid_map = {bssid: int(i + 1) for i, bssid in enumerate(bssid_uniques)}
    dump_pickle("../data/working/bssid_map.pkl", bssid_map)


def encode_bssid(wifi_feature: np.ndarray) -> np.ndarray:
    bssid_map = load_pickle("../data/working/bssid_map.pkl")
    encoded_bssid = [
        [bssid_map[d] for d in row[0]]
        for row in track(wifi_feature, total=len(wifi_feature))
    ]
    wifi_feature[:, 0] = np.array(encoded_bssid)
    return wifi_feature


def wifi_preprocessing():
    # Train data
    filepath = "../data/working/train_wifi_features.npy"
    if not pathlib.Path(filepath).exists():
        with timer("Dump wifi of train"):
            wifi_features = get_wifi_from_waypoints_in_parallel()
            np.save(filepath, wifi_features)

    # Test data
    filepath = "../data/working/test_wifi_features.npy"
    if not pathlib.Path(filepath).exists():
        with timer("Dump wifi of test"):
            wifi_features = get_test_wifi_from_waypoints_in_parallel()
            np.save(filepath, wifi_features)

    # Label encoding
    dump_wifi_bssid_map()
    filepath = "../data/working/train_encoded_wifi_feature.npy"
    if not pathlib.Path(filepath).exists():
        wifi_features = np.load("../data/working/train_wifi_features.npy")
        wifi_features = encode_bssid(wifi_features)
        wifi_features = wifi_features.astype("int64")
        np.save(filepath, wifi_features)

    filepath = "../data/working/test_wifi_encoded_feature.npy"
    if not pathlib.Path(filepath).exists():
        wifi_features = np.load("../data/working/test_wifi_features.npy")
        wifi_features = encode_bssid(wifi_features)
        wifi_features = wifi_features.astype("int64")
        np.save(filepath, wifi_features)


def main():
    print("Processing waypoint ...")
    waypoint_preprocessing()

    print("Processing wifi ...")
    wifi_preprocessing()


if __name__ == "__main__":
    with timer("ParseData"):
        main()
