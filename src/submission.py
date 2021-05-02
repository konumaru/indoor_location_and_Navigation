import pathlib
import statistics
import numpy as np
import pandas as pd
from rich.progress import track
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset, DataLoader

from models import InddorModel
from preprocessing import create_wifi, create_beacon

from utils.feature import FeatureStore
from utils.common import timer
from utils.common import load_pickle, dump_pickle, save_cache


UPDATE_RAW_DATA = False


def extract_raw_data():
    src_dir = pathlib.Path("../data/raw/test/")
    filepaths = [path_filepath for path_filepath in src_dir.glob("*")]

    for filepath in track(filepaths):
        site_id = filepath.parent.parent.name
        floor = "F1"
        path_id = filepath.name.split(".")[0]

        feature = FeatureStore(
            site_id=site_id,
            floor=floor,
            path_id=path_id,
            input_path="../data/raw/",
            save_path="../data/submit/path_data/",
            is_test=True,
        )
        feature.load_all_data()
        feature.save()


def transform_by_scaler_and_save_npy(data: np.ndarray, name: str, seq_len: int):
    scaler = load_pickle(f"../data/scaler/scaler_{name}.pkl")
    data = scaler.transform(data.reshape(-1, 1))
    data = data.astype("float32").reshape(-1, seq_len)
    np.save(f"../data/submit/test_{name}.npy", data)


@save_cache("../data/submit/test_waypoint.pkl", False)
def create_test_waypoint():
    waypoint = pd.read_csv("../data/raw/sample_submission.csv")
    waypoint[["site", "path", "timestamp"]] = waypoint["site_path_timestamp"].str.split(
        "_", expand=True
    )
    waypoint.drop(["site_path_timestamp"], axis=1, inplace=True)

    waypoint = waypoint[["timestamp", "x", "y", "site", "floor", "path"]]

    # Update floor from extend file.
    sub = pd.read_csv("../data/extend/99_acc_submission.csv")
    waypoint["floor"] = sub["floor"]
    return waypoint


@save_cache("../data/submit/test_site_id.pkl", True)
def create_test_build():
    waypoint = load_pickle("../data/submit/test_waypoint.pkl", verbose=False)
    site_map = load_pickle("../data/label_encode/map_site_id.pkl", verbose=False)
    build = waypoint["site"].map(site_map).to_numpy()

    np.save("../data/submit/test_site_id.npy", build)
    return build


@save_cache("../data/submit/test_wifi_results.pkl", True)
def get_wifi_results():
    waypoint = load_pickle("../data/submit/test_waypoint.pkl", verbose=False)
    results = create_wifi(waypoint, "../data/submit/path_data")
    return results


def create_wifi_feature():
    results = get_wifi_results()
    bssid, rssi, freq, last_seen_ts = zip(*results)

    bssid = np.concatenate(bssid, axis=0).astype("int32")
    rssi = np.concatenate(rssi, axis=0)
    freq = np.concatenate(freq, axis=0)
    last_seen_ts = np.concatenate(last_seen_ts, axis=0)

    np.save("../data/submit/test_wifi_bssid.npy", bssid)
    transform_by_scaler_and_save_npy(rssi, "wifi_rssi", 100)
    transform_by_scaler_and_save_npy(freq, "wifi_freq", 100)
    transform_by_scaler_and_save_npy(last_seen_ts, "wifi_last_seen_ts", 100)


@save_cache("../data/submit/test_beacon_results.pkl", True)
def get_beacon_results():
    waypoint = load_pickle("../data/submit/test_waypoint.pkl", verbose=False)
    results = create_beacon(waypoint, "../data/submit/path_data")
    return results


def create_beacon_feature():
    results = get_beacon_results()
    uuid, tx_power, rssi = zip(*results)

    uuid = np.concatenate(uuid, axis=0).astype("int32")
    tx_power = np.concatenate(tx_power, axis=0)
    rssi = np.concatenate(rssi, axis=0)

    np.save("../data/submit/test_beacon_uuid.npy", uuid)
    transform_by_scaler_and_save_npy(tx_power, "beacon_tx_power", 20)
    transform_by_scaler_and_save_npy(rssi, "beacon_rssi", 20)


class IndoorTestDataset(Dataset):
    def __init__(self):
        # Load features.
        featfure_dir = pathlib.Path("../data/submit/")

        wp = load_pickle("../data/submit/test_waypoint.pkl")
        self.floor = wp[["floor"]].to_numpy()

        # Build feature.
        site_id = np.load(featfure_dir / "test_site_id.npy")
        self.site_id = site_id

        # Wifi features.
        wifi_bssid = np.load(featfure_dir / "test_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "test_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "test_wifi_freq.npy")
        wifi_last_seen_ts = np.load(featfure_dir / "test_wifi_last_seen_ts.npy")

        self.wifi_bssid = wifi_bssid
        self.wifi_rssi = wifi_rssi
        self.wifi_freq = wifi_freq
        self.wifi_last_seen_ts = wifi_last_seen_ts

        # Beacon featurees.
        beacon_uuid = np.load(featfure_dir / "test_beacon_uuid.npy")
        beacon_tx_power = np.load(featfure_dir / "test_beacon_tx_power.npy")
        beacon_rssi = np.load(featfure_dir / "test_beacon_rssi.npy")

        self.beacon_uuid = beacon_uuid
        self.beacon_tx_power = beacon_tx_power
        self.beacon_rssi = beacon_rssi

    def __len__(self):
        return len(self.site_id)

    def __getitem__(self, idx):
        x_build = (self.site_id[idx], self.floor[idx])
        x_wifi = (
            self.site_id[idx],
            self.wifi_bssid[idx],
            self.wifi_rssi[idx],
            self.wifi_freq[idx],
            self.wifi_last_seen_ts[idx],
        )
        x_beacon = (
            self.beacon_uuid[idx],
            self.beacon_tx_power[idx],
            self.beacon_rssi[idx],
        )
        x = (x_build, x_wifi, x_beacon)
        return x


def load_checkpoints(model_name: str):
    with open(f"../checkpoints/{model_name}.txt", "r") as f:
        checkpoints = f.readlines()
        checkpoints = [c.strip() for c in checkpoints]
    return checkpoints


def main():
    # Extract raw data.
    if UPDATE_RAW_DATA:
        extract_raw_data()

    # Create preprocessing data.
    _ = create_test_waypoint()
    _ = create_test_build()
    _ = create_wifi_feature()
    _ = create_beacon_feature()

    # Define dataset and dataloader.
    dataset = IndoorTestDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Load model and predict.
    checkpoints = load_checkpoints("CV-GroupKfold5")
    floor = []
    postion = []
    for _, ckpt in enumerate(track(checkpoints)):
        model = InddorModel.load_from_checkpoint(ckpt)
        model.eval()
        model.freeze()

        _floor = []
        _postion = []
        for batch in dataloader:
            floor_hat, pos_hat = model(batch)

            # _floor.append(y_hat[0])
            _postion.append(pos_hat)

        # _floor = torch.cat(_floor, dim=0).detach().numpy().copy()
        _postion = torch.cat(_postion, dim=0).detach().cpu().numpy()

        # floor.append(_floor)
        postion.append(_postion)

    # floor = (
    #     pd.DataFrame(np.concatenate(floor, axis=1))
    #     .apply(lambda x: statistics.mode(x), axis=1)
    #     .to_numpy()
    # )
    postion = np.mean(postion, axis=0)

    # Dump submission file.
    submission = pd.read_csv("../data/raw/sample_submission.csv")
    # submission.iloc[:, 1] = floor
    submission.iloc[:, 2:] = postion
    print(submission.head())
    print(submission["floor"].value_counts().sort_index())

    submission["floor"] = submission["floor"].astype(int)
    submission.to_csv("../data/submit/submission.csv", index=False)

    assert submission.isnull().mean().max() == 0.0


if __name__ == "__main__":
    with timer("Submission"):
        main()
