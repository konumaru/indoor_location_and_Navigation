import pathlib
import numpy as np
import pandas as pd
from rich.progress import track
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset, DataLoader

from model import InddorModel
from utils.feature import FeatureStore
from utils.common import load_pickle, dump_pickle, save_cache

UPDATE_RAW_DATA = False


def extract_raw_data():
    src_dir = pathlib.Path("../data/raw/test/")
    filepaths = [path_filepath for path_filepath in src_dir.glob("*")]
    print(str(src_dir))

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


@save_cache("../data/submit/test_wifi_results.pkl", True)
def create_test_wifi():
    def get_wifi_feature(path_id, gdf):
        seq_len = 100
        bssid = []
        rssi = []
        freq = []

        feature = load_pickle(f"../data/submit/path_data/{path_id}.pkl", verbose=False)
        wifi = feature.wifi.copy()
        wifi["bssid"] = wifi["bssid"].map(bssid_map)

        min_idx = gdf.index.min()
        max_idx = gdf.index.max()

        for i, row in gdf.iterrows():
            ts_pre_wp = int(gdf.loc[i - 1, "timestamp"]) if i > min_idx else None
            ts_current_wp = int(gdf.loc[i, "timestamp"])
            ts_post_wp = int(gdf.loc[i + 1, "timestamp"]) if (i + 1) < max_idx else None

            _wifi = wifi.copy()
            # NOTE: ターゲットとなるwaypointとその前後のwaypointの間にあるデータを取得する。
            ts_wifi = _wifi["timestamp"].values
            pre_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_pre_wp is None
                else (ts_pre_wp < ts_wifi)
            )
            psot_flag = (
                np.ones(len(ts_wifi)).astype(bool)
                if ts_post_wp is None
                else (ts_wifi < ts_post_wp)
            )
            _wifi = _wifi[pre_flag & psot_flag]

            _wifi = _wifi.sort_values(by="rssi", ascending=False)
            _wifi = _wifi.head(seq_len)

            _bssid = np.zeros(seq_len)
            _rssi = np.tile(-999, seq_len)
            _freq = np.tile(-999, seq_len)

            _bssid[: len(_wifi)] = _wifi["bssid"].fillna(0).to_numpy()
            _rssi[: len(_wifi)] = _wifi["rssi"].fillna(-999).to_numpy()
            _freq[: len(_wifi)] = _wifi["frequency"].fillna(-999).to_numpy()

            bssid.append(_bssid.astype("int32"))
            rssi.append(_rssi.astype("float32"))
            freq.append(_freq.astype("float32"))

        return bssid, rssi, freq

    waypoint = load_pickle("../data/submit/test_waypoint.pkl", verbose=False)
    bssid_map = load_pickle("../data/label_encode/map_bssid.pkl", verbose=False)
    results = Parallel(n_jobs=-1)(
        delayed(get_wifi_feature)(path_id, gdf)
        for path_id, gdf in track(waypoint.groupby("path"))
    )
    return results


def create_wifi_feature():
    results = create_test_wifi()
    bssid, rssi, freq = zip(*results)

    bssid = np.concatenate(bssid, axis=0).astype("int32")
    np.save("../data/submit/test_wifi_bssid.npy", bssid)

    scaler = load_pickle("../data/scaler/scaler_rssi.pkl")
    rssi = np.concatenate(rssi, axis=0)
    rssi = scaler.transform(rssi)
    rssi = rssi.astype("float32")
    np.save("../data/submit/test_wifi_rssi.npy", rssi)

    scaler = load_pickle("../data/scaler/scaler_rssi.pkl")
    freq = np.concatenate(freq, axis=0)
    freq = scaler.transform(freq)
    freq = freq.astype("float32")
    np.save("../data/submit/test_wifi_freq.npy", freq)


class IndoorTestDataset(Dataset):
    def __init__(self):
        # Load features.
        featfure_dir = pathlib.Path("../data/submit/")

        site_id = np.load(featfure_dir / "test_site_id.npy")

        wifi_bssid = np.load(featfure_dir / "test_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "test_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "test_wifi_freq.npy")

        self.site_id = site_id.reshape(-1, 1)

        self.wifi_bssid = wifi_bssid[:, :20]
        self.wifi_rssi = wifi_rssi[:, :20]
        self.wifi_freq = wifi_freq[:, :20]

    def __len__(self):
        return len(self.site_id)

    def __getitem__(self, idx):
        x_build = self.site_id[idx]
        x_wifi = (
            self.wifi_bssid[idx],
            self.wifi_rssi[idx],
            self.wifi_freq[idx],
        )
        x = (x_build, x_wifi)
        return x


def main():
    # Extract raw data.
    if UPDATE_RAW_DATA:
        extract_raw_data()

    # Create preprocessing data.
    _ = create_test_waypoint()
    _ = create_test_build()
    _ = create_wifi_feature()

    # Define dataset and dataloader.
    dataset = IndoorTestDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        drop_last=False,
    )

    # Load model and predict.
    model = InddorModel.load_from_checkpoint(
        "../tb_logs/Baseline/version_10/checkpoints/epoch=5-step=1559.ckpt"
    )
    model.eval()
    model.freeze()

    preds = []
    for batch in dataloader:
        y_hat = model(batch)
        preds.append(y_hat)

    pred = torch.cat(preds, dim=0)
    pred = pred.detach().numpy().copy()
    print(pred.shape)

    # Dump submission file.
    submission = pd.read_csv("../data/raw/sample_submission.csv")
    submission.iloc[:, 1:] = pred
    print(submission.head())

    submission["floor"] = submission["floor"].astype(int)
    submission.to_csv("../data/submit/submission.csv", index=False)

    assert submission.isnull().mean().max() == 0.0

    print(submission["floor"].value_counts().sort_index())


if __name__ == "__main__":
    main()
