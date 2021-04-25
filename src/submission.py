import pathlib
import numpy as np
import pandas as pd
from rich.progress import track
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset, DataLoader

from models import InddorModel
from preprocessing import create_wifi, create_beacon
from utils.feature import FeatureStore
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


def transform_by_scaler_and_save_npy(data: np.ndarray, name: str):
    scaler = load_pickle(f"../data/scaler/scaler_{name}.pkl")
    data = scaler.transform(data)
    data = data.astype("float32")
    np.save(f"../data/submit/test_{name}.npy", data)


@save_cache("../data/submit/test_waypoint.pkl", True)
def create_test_waypoint():
    waypoint = pd.read_csv("../data/raw/sample_submission.csv")
    waypoint[["site", "path", "timestamp"]] = waypoint["site_path_timestamp"].str.split(
        "_", expand=True
    )
    waypoint.drop(["site_path_timestamp"], axis=1, inplace=True)

    waypoint = waypoint[["timestamp", "x", "y", "site", "floor", "path"]]
    return waypoint


@save_cache("../data/submit/test_site_id.pkl", True)
def create_test_build():
    def get_waypoint_from_featureStore(path_id):
        feature = load_pickle(f"../data/submit/path_data/{path_id}.pkl", verbose=False)
        return site_map[feature.site_id]

    waypoint = load_pickle("../data/submit/test_waypoint.pkl", verbose=False)
    site_map = load_pickle("../data/label_encode/map_site_id.pkl", verbose=False)
    build = waypoint["site"].map(site_map).to_numpy()

    np.save("../data/submit/test_site_id.npy", build)
    return build


@save_cache("../data/submit/test_wifi_results.pkl", False)
def get_wifi_results():
    waypoint = load_pickle("../data/submit/test_waypoint.pkl", verbose=False)
    results = create_wifi(waypoint, "../data/submit/path_data")
    return results


def create_wifi_feature():
    results = get_wifi_results()
    bssid, rssi, freq = zip(*results)

    bssid = np.concatenate(bssid, axis=0).astype("int32")
    rssi = np.concatenate(rssi, axis=0)
    freq = np.concatenate(freq, axis=0)

    np.save(f"../data/submit/test_wifi_bssid.npy", bssid)
    transform_by_scaler_and_save_npy(rssi, "wifi_rssi")
    transform_by_scaler_and_save_npy(freq, "wifi_freq")


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

    np.save(f"../data/submit/test_beacon_uuid.npy", uuid)
    transform_by_scaler_and_save_npy(tx_power, "beacon_tx_power")
    transform_by_scaler_and_save_npy(rssi, "beacon_rssi")


class IndoorTestDataset(Dataset):
    def __init__(self):
        # Load features.
        featfure_dir = pathlib.Path("../data/submit/")

        # Build feature.
        site_id = np.load(featfure_dir / "test_site_id.npy")
        self.site_id = site_id.reshape(-1, 1)

        # Wifi features.
        wifi_bssid = np.load(featfure_dir / "test_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "test_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "test_wifi_freq.npy")

        self.wifi_bssid = wifi_bssid[:, :20]
        self.wifi_rssi = wifi_rssi[:, :20]
        self.wifi_freq = wifi_freq[:, :20]

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
        x_build = self.site_id[idx]
        x_wifi = (
            self.wifi_bssid[idx],
            self.wifi_rssi[idx],
            self.wifi_freq[idx],
        )
        x_beacon = (
            self.beacon_uuid[idx],
            self.beacon_tx_power[idx],
            self.beacon_rssi[idx],
        )
        x = (x_build, x_wifi, x_beacon)
        return x


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
        shuffle=False,
        drop_last=False,
    )

    # Load model and predict.
    checkpoint = (
        "tb_logs/Recreate-labelencodemap/version_2/checkpoints/epoch=18-step=5566.ckpt"
    )
    model = InddorModel.load_from_checkpoint(f"../{checkpoint}")
    model.eval()
    model.freeze()

    floor = []
    postion = []
    for batch in dataloader:
        y_hat = model(batch)

        floor.append(y_hat[0])
        postion.append(y_hat[1])

    floor = torch.cat(floor, dim=0)
    postion = torch.cat(postion, dim=0)

    floor = floor.detach().numpy().copy().ravel()
    postion = postion.detach().numpy().copy()

    # Dump submission file.
    submission = pd.read_csv("../data/raw/sample_submission.csv")
    submission.iloc[:, 1] = floor
    submission.iloc[:, 2:] = postion
    print(submission.head())
    print(submission["floor"].value_counts().sort_index())

    submission["floor"] = submission["floor"].astype(int)
    submission.to_csv("../data/submit/submission.csv", index=False)

    assert submission.isnull().mean().max() == 0.0


if __name__ == "__main__":
    main()
