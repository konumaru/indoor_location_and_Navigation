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
            ts_threshold = 3000
            ts_diff = np.abs(wifi["timestamp"].astype(int) - int(row["timestamp"]))
            _wifi = wifi[ts_diff <= ts_threshold].copy()

            _wifi.sort_values(by=["rssi"], ascending=False, inplace=True)
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


@save_cache("../data/preprocessing/train_wifi_results.pkl", True)
def get_wifi_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = create_wifi(waypoint)
    return results


def create_wifi_feature():
    results = get_wifi_results()
    bssid, rssi, freq, last_seen_ts = zip(*results)

    def save_scaler_and_npy(data: np.ndarray, name: str):
        scaler = StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, 1))
        data = data.astype("float32").reshape(-1, 100)
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
            ts_threshold = 3000
            ts_diff = np.abs(data["timestamp"].astype(int) - int(row["timestamp"]))
            _data = data[ts_diff <= ts_threshold].copy()

            _data.sort_values(by=["rssi"], ascending=False, inplace=True)
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
        data = scaler.fit_transform(data.reshape(-1, 1))
        data = data.astype("float32").reshape(-1, 20)
        dump_pickle(f"../data/scaler/scaler_{name}.pkl", scaler)
        np.save(f"../data/preprocessing/train_{name}.npy", data)

    uuid = np.concatenate(uuid, axis=0).astype("int32")
    tx_power = np.concatenate(tx_power, axis=0)
    rssi = np.concatenate(rssi, axis=0)

    np.save("../data/preprocessing/train_beacon_uuid.npy", uuid)
    save_scaler_and_npy(tx_power, "beacon_tx_power")
    save_scaler_and_npy(rssi, "beacon_rssi")


# === common sensor data parser. ===


def get_sensor_feature(
    path_id: str,
    gdf: pd.DataFrame,
    data_name: str = "accelerometer",
    fill_value: float = -99.0,
):
    past_X, past_Y, past_Z = [], [], []
    feat_X, feat_Y, feat_Z = [], [], []

    feature = load_pickle(f"../data/working/{path_id}.pkl", verbose=False)
    data = feature[data_name]

    seq_len = 100
    data_size = gdf.shape[0]
    past_x = np.tile(fill_value, (data_size, seq_len))
    past_y = np.tile(fill_value, (data_size, seq_len))
    past_z = np.tile(fill_value, (data_size, seq_len))

    feat_x = np.tile(fill_value, (data_size, seq_len))
    feat_y = np.tile(fill_value, (data_size, seq_len))
    feat_z = np.tile(fill_value, (data_size, seq_len))

    for i, (_, row) in enumerate(gdf.iterrows()):
        ts_wp = row["timestamp"]
        # Past features.
        _data = data.loc[data["timestamp"] < ts_wp, ["x", "y", "z"]].tail(seq_len)
        _data_size = _data.shape[0]

        past_x[i, :_data_size] = _data["x"]
        past_y[i, :_data_size] = _data["y"]
        past_z[i, :_data_size] = _data["z"]
        # Feature features.
        _data = data.loc[data["timestamp"] > ts_wp, ["x", "y", "z"]].head(seq_len)
        _data_size = _data.shape[0]

        feat_x[i, :_data_size] = _data["x"]
        feat_y[i, :_data_size] = _data["y"]
        feat_z[i, :_data_size] = _data["z"]

    past_X.append(np.fliplr(past_x))
    past_Y.append(np.fliplr(past_y))
    past_Z.append(np.fliplr(past_z))

    feat_X.append(np.fliplr(feat_x))
    feat_Y.append(np.fliplr(feat_y))
    feat_Z.append(np.fliplr(feat_z))

    return past_X, past_Y, past_Z, feat_X, feat_Y, feat_Z


def save_scaler_and_npy(data: np.ndarray, name: str, seq_len: int = 100):
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))
    data = data.astype("float32").reshape(-1, seq_len)
    dump_pickle(f"../data/scaler/scaler_{name}.pkl", scaler)
    np.save(f"../data/preprocessing/train_{name}.npy", data)


# === accelerometer


@save_cache("../data/preprocessing/train_acce_results.pkl", False)
def get_acce_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_sensor_feature)(path_id, gdf, "accelerometer", -10.0)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


def create_accelerometer_feature():
    results = get_acce_results()
    past_X, past_Y, past_Z, feat_X, feat_Y, feat_Z = zip(*results)

    past_X = np.concatenate(past_X, axis=1).reshape(-1, 100)
    past_Y = np.concatenate(past_Y, axis=1).reshape(-1, 100)
    past_Z = np.concatenate(past_Z, axis=1).reshape(-1, 100)

    feat_X = np.concatenate(feat_X, axis=1).reshape(-1, 100)
    feat_Y = np.concatenate(feat_Y, axis=1).reshape(-1, 100)
    feat_Z = np.concatenate(feat_Z, axis=1).reshape(-1, 100)

    feature_name = "acce"
    save_scaler_and_npy(past_X, f"{feature_name}_past_X", 100)
    save_scaler_and_npy(past_Y, f"{feature_name}_past_Y", 100)
    save_scaler_and_npy(past_Z, f"{feature_name}_past_Z", 100)
    save_scaler_and_npy(feat_X, f"{feature_name}_feat_X", 100)
    save_scaler_and_npy(feat_Y, f"{feature_name}_feat_Y", 100)
    save_scaler_and_npy(feat_Z, f"{feature_name}_feat_Z", 100)


# === gyroscope ===


@save_cache("../data/preprocessing/train_gyroscope_results.pkl", False)
def get_gyroscope_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_sensor_feature)(path_id, gdf, "gyroscope", -5.0)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


def create_gyroscope_feature():
    results = get_gyroscope_results()
    past_X, past_Y, past_Z, feat_X, feat_Y, feat_Z = zip(*results)

    past_X = np.concatenate(past_X, axis=1).reshape(-1, 100)
    past_Y = np.concatenate(past_Y, axis=1).reshape(-1, 100)
    past_Z = np.concatenate(past_Z, axis=1).reshape(-1, 100)

    feat_X = np.concatenate(feat_X, axis=1).reshape(-1, 100)
    feat_Y = np.concatenate(feat_Y, axis=1).reshape(-1, 100)
    feat_Z = np.concatenate(feat_Z, axis=1).reshape(-1, 100)

    feature_name = "gyroscope"
    save_scaler_and_npy(past_X, f"{feature_name}_past_X", 100)
    save_scaler_and_npy(past_Y, f"{feature_name}_past_Y", 100)
    save_scaler_and_npy(past_Z, f"{feature_name}_past_Z", 100)
    save_scaler_and_npy(feat_X, f"{feature_name}_feat_X", 100)
    save_scaler_and_npy(feat_Y, f"{feature_name}_feat_Y", 100)
    save_scaler_and_npy(feat_Z, f"{feature_name}_feat_Z", 100)


# === magnetic_field ===


@save_cache("../data/preprocessing/train_magnetic_field_results.pkl", False)
def get_magnetic_field_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_sensor_feature)(path_id, gdf, "magnetic_field", -99.0)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


def create_magnetic_feild_feature():
    results = get_gyroscope_results()
    past_X, past_Y, past_Z, feat_X, feat_Y, feat_Z = zip(*results)

    past_X = np.concatenate(past_X, axis=1).reshape(-1, 100)
    past_Y = np.concatenate(past_Y, axis=1).reshape(-1, 100)
    past_Z = np.concatenate(past_Z, axis=1).reshape(-1, 100)

    feat_X = np.concatenate(feat_X, axis=1).reshape(-1, 100)
    feat_Y = np.concatenate(feat_Y, axis=1).reshape(-1, 100)
    feat_Z = np.concatenate(feat_Z, axis=1).reshape(-1, 100)

    feature_name = "magnetic_field"
    save_scaler_and_npy(past_X, f"{feature_name}_past_X", 100)
    save_scaler_and_npy(past_Y, f"{feature_name}_past_Y", 100)
    save_scaler_and_npy(past_Z, f"{feature_name}_past_Z", 100)
    save_scaler_and_npy(feat_X, f"{feature_name}_feat_X", 100)
    save_scaler_and_npy(feat_Y, f"{feature_name}_feat_Y", 100)
    save_scaler_and_npy(feat_Z, f"{feature_name}_feat_Z", 100)


# === rotation_vector ===


@save_cache("../data/preprocessing/train_rotation_vector_results.pkl", False)
def get_rotation_vector_results():
    waypoint = load_pickle("../data/preprocessing/train_waypoint.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_sensor_feature)(path_id, gdf, "rotation_vector", -3.0)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


def create_rotation_vector_feature():
    results = get_rotation_vector_results()
    past_X, past_Y, past_Z, feat_X, feat_Y, feat_Z = zip(*results)

    past_X = np.concatenate(past_X, axis=1).reshape(-1, 100)
    past_Y = np.concatenate(past_Y, axis=1).reshape(-1, 100)
    past_Z = np.concatenate(past_Z, axis=1).reshape(-1, 100)

    feat_X = np.concatenate(feat_X, axis=1).reshape(-1, 100)
    feat_Y = np.concatenate(feat_Y, axis=1).reshape(-1, 100)
    feat_Z = np.concatenate(feat_Z, axis=1).reshape(-1, 100)

    feature_name = "rotation_vector"
    save_scaler_and_npy(past_X, f"{feature_name}_past_X", 100)
    save_scaler_and_npy(past_Y, f"{feature_name}_past_Y", 100)
    save_scaler_and_npy(past_Z, f"{feature_name}_past_Z", 100)
    save_scaler_and_npy(feat_X, f"{feature_name}_feat_X", 100)
    save_scaler_and_npy(feat_Y, f"{feature_name}_feat_Y", 100)
    save_scaler_and_npy(feat_Z, f"{feature_name}_feat_Z", 100)


def main():
    print("\nCreate waypoint ...")
    _ = create_waypoint()

    print("\nCreate build ...")
    _ = create_build_feature()

    print("\nCreate wifi ...")
    _ = create_wifi_feature()

    print("\nCreate beacon ...")
    _ = create_beacon_feature()

    print("\nCreate accelerometer ...")
    _ = create_accelerometer_feature()

    print("\nCreate gyroscope ...")
    _ = create_gyroscope_feature()

    print("\nCreate magnetic_feild ...")
    _ = create_magnetic_feild_feature()

    print("\nCreate rotation_vector ...")
    _ = create_rotation_vector_feature()


if __name__ == "__main__":
    with timer("Creat Features"):
        main()
