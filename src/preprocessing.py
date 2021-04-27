import pathlib
import numpy as np
import pandas as pd
from rich.progress import track
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler

from typing import List

from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache
from utils.feature import FeatureStore

# == waypoint ===


@save_cache("../data/preprocessing/train_waypoint.pkl", True)
def create_waypoint():
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

    src_dir = pathlib.Path("../data/raw/train/")
    filepaths = [
        path_filepath
        for site_filepath in src_dir.glob("*")
        for floor_filepath in site_filepath.glob("*")
        for path_filepath in floor_filepath.glob("*")
    ]

    waypoint = Parallel(n_jobs=-1)(
        delayed(get_waypoint_from_featureStore)(filepath)
        for filepath in track(filepaths)
    )
    waypoint = pd.concat(waypoint, axis=0).reset_index(drop=True)
    waypoint = waypoint.sort_values(by=["path", "timestamp"]).reset_index(drop=True)
    return waypoint


# === build ===


@save_cache("../data/preprocessing/train_build_results.pkl", True)
def create_build():
    def get_waypoint_from_featureStore(path_id):
        feature = load_pickle(f"../data/working/{path_id}.pkl", verbose=False)
        return (
            site_map[feature.site_id],
            feature.site_info.site_height,
            feature.site_info.site_width,
        )

    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    site_map = load_pickle("../data/label_encode/map_site_id.pkl", verbose=False)
    resutls = Parallel(n_jobs=-1)(
        delayed(get_waypoint_from_featureStore)(filepath)
        for filepath in track(waypoint["path"].values)
    )
    return resutls


def create_build_feature():
    results = create_build()
    site_id, site_height, site_width = zip(*results)

    site_id = np.array(site_id, dtype="int32")
    np.save("../data/preprocessing/train_site_id.npy", site_id)

    site_height = np.array(site_height, dtype="float32")
    np.save("../data/preprocessing/train_site_height.npy", site_height)

    site_width = np.array(site_width, dtype="float32")
    np.save("../data/preprocessing/train_site_width.npy", site_width)


# ==== wifi ===


def create_wifi(waypoint: np.ndarray, scr_dir: str = "../data/working"):
    def get_wifi_feature(path_id, gdf):
        ts_waypoint = gdf["timestamp"].to_numpy()
        min_idx = 0
        max_idx = len(ts_waypoint) - 1

        feature = load_pickle(f"{scr_dir}/{path_id}.pkl", verbose=False)
        wifi = feature.wifi.copy()
        wifi["bssid"] = wifi["bssid"].map(bssid_map).astype("int32")
        # wifi.loc[~wifi["bssid"].isin(bssid_map.keys()), "bssid"] = 0

        seq_len = 100
        bssid = []
        rssi = []
        freq = []
        last_seen_ts = []

        for i, (_, row) in enumerate(gdf.iterrows()):
            n_diff = 3
            ts_pre_wp = ts_waypoint[i - n_diff] if (i - n_diff) >= min_idx else None
            ts_post_wp = ts_waypoint[i + n_diff] if (i + n_diff) <= max_idx else None
            # NOTE: ターゲットとなるwaypointとその前後のwaypointの間にあるデータを取得する。
            ts_wifi = wifi["timestamp"].to_numpy()
            pre_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_pre_wp is None
                else (int(ts_pre_wp) < ts_wifi)
            )
            psot_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_post_wp is None
                else (ts_wifi < int(ts_post_wp))
            )
            _wifi = wifi[pre_flag & psot_flag].copy()
            _wifi.sort_values(by="rssi", ascending=False, inplace=True)
            _wifi.drop_duplicates(
                subset=["bssid", "rssi", "frequency", "last_seen_timestamp"],
                keep="first",
                inplace=True,
            )
            _wifi = _wifi.head(seq_len)
            # Fillna
            _wifi["bssid"].fillna(0, inplace=True)
            _wifi["rssi"].fillna(-999, inplace=True)
            _wifi["frequency"].fillna(-999, inplace=True)
            _wifi["last_seen_timestamp"].fillna(-999, inplace=True)

            _bssid = np.zeros(seq_len)
            _rssi = np.tile(-999, seq_len)
            _freq = np.tile(-999, seq_len)
            _last_seen_ts = np.tile(-999, seq_len)

            _bssid[: len(_wifi)] = _wifi["bssid"].astype("int32").to_numpy()
            _rssi[: len(_wifi)] = _wifi["rssi"].astype("float32").to_numpy()
            _freq[: len(_wifi)] = _wifi["frequency"].astype("float32").to_numpy()
            _last_seen_ts[: len(_wifi)] = (
                _wifi["last_seen_timestamp"].astype("float32").to_numpy()
            )

            bssid.append(_bssid)
            rssi.append(_rssi)
            freq.append(_freq)
            last_seen_ts.append(_last_seen_ts)

        return bssid, rssi, freq, last_seen_ts

    bssid_map = load_pickle("../data/label_encode/map_bssid.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_wifi_feature)(path_id, gdf)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


@save_cache("../data/preprocessing/train_wifi_results.pkl", False)
def get_wifi_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = create_wifi(waypoint)
    return results


def create_wifi_feature():
    results = get_wifi_results()
    bssid, rssi, freq, last_seen_ts = zip(*results)

    def save_scaler_and_npy(data: np.ndarray, name: str):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = data.astype("float32")
        dump_pickle(f"../data/scaler/scaler_{name}.pkl", scaler)
        np.save(f"../data/preprocessing/train_{name}.npy", data)

    bssid = np.concatenate(bssid, axis=0).astype("int32")
    rssi = np.concatenate(rssi, axis=0)
    freq = np.concatenate(freq, axis=0)
    last_seen_ts = np.concatenate(last_seen_ts, axis=0)

    np.save("../data/preprocessing/train_wifi_bssid.npy", bssid)
    save_scaler_and_npy(rssi, "wifi_rssi")
    save_scaler_and_npy(freq, "wifi_freq")
    save_scaler_and_npy(last_seen_ts, "wifi_last_seen_ts")


# === beacon ===


def create_beacon(waypoint: np.ndarray, scr_dir: str = "../data/working"):
    def get_beacon_feature(path_id, gdf):
        ts_waypoint = gdf["timestamp"].to_numpy()
        min_idx = 0
        max_idx = len(ts_waypoint) - 1

        feature = load_pickle(f"{scr_dir}/{path_id}.pkl", verbose=False)
        data = feature.beacon.copy()
        data["uuid"] = data["uuid"].map(uuid_map)

        seq_len = 20
        uuid = []
        tx_power = []
        rssi = []

        for i, (_, row) in enumerate(gdf.iterrows()):
            n_diff = 1
            ts_pre_wp = ts_waypoint[i - n_diff] if (i - n_diff) >= min_idx else None
            ts_post_wp = ts_waypoint[i + n_diff] if (i + n_diff) <= max_idx else None

            # NOTE: ターゲットとなるwaypointとその前後のwaypointの間にあるデータを取得する。
            ts_data = data["timestamp"].to_numpy()
            pre_flag = (
                np.ones(len(ts_data)).astype(bool)
                if ts_pre_wp == None
                else (int(ts_pre_wp) < ts_data)
            )
            psot_flag = (
                np.ones(len(ts_data)).astype(bool)
                if ts_post_wp == None
                else (ts_data < int(ts_post_wp))
            )

            _data = data[pre_flag & psot_flag].copy()
            _data.sort_values(by="rssi", ascending=False, inplace=True)
            _data.drop_duplicates(
                subset=["uuid", "tx_power", "rssi"], keep="first", inplace=True
            )
            _data = _data.head(seq_len)

            _uuid = np.zeros(seq_len)
            _tx_power = np.tile(-999, seq_len)
            _rssi = np.tile(-999, seq_len)

            _uuid[: len(_data)] = _data["uuid"].fillna(0).astype("int32").to_numpy()
            _tx_power[: len(_data)] = (
                _data["tx_power"].fillna(-999).astype("float32").to_numpy()
            )
            _rssi[: len(_data)] = (
                _data["rssi"].fillna(-999).astype("float32").to_numpy()
            )

            uuid.append(_uuid)
            tx_power.append(_tx_power)
            rssi.append(_rssi)

        return uuid, tx_power, rssi

    uuid_map = load_pickle("../data/label_encode/map_beacon_uuid.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_beacon_feature)(path_id, gdf)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


@save_cache("../data/preprocessing/train_beacon_results.pkl", True)
def get_beacon_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = create_beacon(waypoint)
    return results


def create_beacon_feature():
    results = get_beacon_results()
    uuid, tx_power, rssi = zip(*results)

    def save_scaler_and_npy(data: np.ndarray, name: str):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = data.astype("float32")
        dump_pickle(f"../data/scaler/scaler_{name}.pkl", scaler)
        np.save(f"../data/preprocessing/train_{name}.npy", data)

    uuid = np.concatenate(uuid, axis=0).astype("int32")
    tx_power = np.concatenate(tx_power, axis=0)
    rssi = np.concatenate(rssi, axis=0)

    np.save("../data/preprocessing/train_beacon_uuid.npy", uuid)
    save_scaler_and_npy(tx_power, "beacon_tx_power")
    save_scaler_and_npy(rssi, "beacon_rssi")


# === accelerometer


def create_accelerometer_feature():
    return None


# === gyroscope ===


def create_gyroscope_feature():
    return None


# === magnetic_field ===


def create_magnetic_feild_feature():
    return None


# === rotation_vector ===


def create_rotation_vector_feature():
    return None


def main():
    print("\nCreate waypoint ...")
    _ = create_waypoint()

    print("\nCreate build ...")
    _ = create_build_feature()

    print("\nCreate wifi ...")
    _ = create_wifi_feature()

    print("\nCreate beacon ...")
    _ = create_beacon_feature()


if __name__ == "__main__":
    with timer("Creat Features"):
        main()
