import pathlib
import numpy as np
import numpy.typing as npt
from rich.progress import track

from utils import timer
from utils import load_pickle, dump_pickle, save_cache

from typing import List, Dict


@save_cache("../data/preprocessing/train_data_flag.pkl", True)
def create_use_data_flag() -> np.ndarray:
    waypoint = load_pickle("../data/working/train_waypint.pkl")
    floor = waypoint[:, 1]
    use_floor = np.array(
        ["B3", "B2", "B1", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
    )
    use_flag = np.in1d(floor, use_floor)
    return use_flag


# === ids, (site, floor, path), string ===
@save_cache("../data/preprocessing/train_ids.pkl", True)
def create_train_ids() -> np.ndarray:
    waypoint = load_pickle("../data/working/train_waypint.pkl")
    # Select (site, floor, path) from (site, floor, path. timestamp, x, y)
    ids = waypoint[:, [0, 1, 2]]
    flag = load_pickle("../data/preprocessing/train_data_flag.pkl")
    ids = ids[flag]
    return ids


# === build, (site, floor, path), string ===
@save_cache("../data/preprocessing/site_map.pkl", True)
def create_site_map() -> Dict:
    waypoint_train = load_pickle("../data/working/train_waypint.pkl")
    waypoint_test = load_pickle("../data/working/test_waypint.pkl")
    # wifi columns is (bssid, rssi, frequency).
    site_uniques = np.unique(
        np.concatenate([waypoint_train[:, 0], waypoint_test[:, 0]], axis=0).ravel()
    )
    site_map = {bssid: int(i + 1) for i, bssid in enumerate(site_uniques)}
    return site_map


def encode_site(ids: np.ndarray) -> np.ndarray:
    site_map = load_pickle("../data/preprocessing/site_map.pkl")
    encoded_site = [site_map[row[0]] for row in ids]
    ids[:, 0] = np.array(encoded_site)
    return ids


@save_cache("../data/preprocessing/train_build.pkl", True)
def create_train_build() -> np.ndarray:
    waypoint = load_pickle("../data/working/train_waypint.pkl")
    # Select (site, floor, path) from (site, floor, path. timestamp, x, y)
    ids = waypoint[:, [0, 1, 2]]
    # Label encode for site.
    ids = encode_site(ids)
    # Filtering data.
    flag = load_pickle("../data/preprocessing/train_data_flag.pkl")
    ids = ids[flag]
    ids = ids[:, 0].astype("int64").reshape(-1, 1)
    return ids


# === target, (floor, x, y, is_encoded), string ===


@save_cache("../data/preprocessing/train_target.pkl", True)
def create_train_target() -> np.ndarray:
    waypoint = load_pickle("../data/working/train_waypint.pkl")
    # Select (floor, x, y) from (site, floor, path. timestamp, x, y)
    target = waypoint[:, [1, 4, 5]]
    # Label encode for floor
    floorNums = {
        "B3": -3,
        "B2": -2,
        "B1": -1,
        "F1": 0,
        "F2": 1,
        "F3": 2,
        "F4": 3,
        "F5": 4,
        "F6": 5,
        "F7": 6,
        "F8": 7,
        "F9": 8,
        "F10": 9,
    }
    for key, val in floorNums.items():
        target[:, 0] = np.char.replace(target[:, 0], key, str(val))

    # Filtering data
    flag = load_pickle("../data/preprocessing/train_data_flag.pkl")
    target = target[flag].astype("float64")
    return target


# === wifi===


@save_cache("../data/preprocessing/bssid_map.pkl", True)
def create_bssid_map() -> Dict:
    wifi_train = load_pickle("../data/working/train_wifi.pkl")
    wifi_test = load_pickle("../data/working/test_wifi.pkl")
    # wifi columns is (bssid, rssi, frequency).
    bssid_uniques = np.unique(
        np.concatenate([wifi_train[:, 0], wifi_test[:, 0]], axis=0).ravel()
    )
    bssid_map = {bssid: int(i + 1) for i, bssid in enumerate(bssid_uniques)}
    return bssid_map


def encode_bssid(wifi_feature: np.ndarray) -> np.ndarray:
    bssid_map = load_pickle("../data/preprocessing/bssid_map.pkl")
    encoded_bssid = [[bssid_map[d] for d in row[0]] for row in wifi_feature]
    wifi_feature[:, 0] = np.array(encoded_bssid)
    return wifi_feature


@save_cache("../data/preprocessing/train_wifi.pkl", True)
def create_train_wifi() -> np.ndarray:
    wifi = load_pickle("../data/working/train_wifi.pkl")
    wifi = encode_bssid(wifi)
    # Filtering data.
    flag = load_pickle("../data/preprocessing/train_data_flag.pkl")
    wifi = wifi[flag].astype("int64")
    return wifi


@save_cache("../data/preprocessing/test_wifi.pkl", True)
def create_test_wifi() -> np.ndarray:
    wifi = load_pickle("../data/working/train_wifi.pkl")
    wifi = encode_bssid(wifi)
    wifi = wifi.astype("int64")
    return wifi


def main():
    # Dump flag.
    print("Create data flag ...")
    _ = create_use_data_flag()
    _ = create_train_ids()

    print("\nCreate target ...")
    _ = create_train_target()

    print("\nCreate build ...")
    _ = create_site_map()
    _ = create_train_build()

    print("\nCreate wifi ...")
    _ = create_bssid_map()
    _ = create_train_wifi()
    _ = create_test_wifi()


if __name__ == "__main__":
    main()
