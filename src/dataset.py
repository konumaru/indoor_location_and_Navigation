import pathlib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from utils.common import load_pickle


class IndoorDataset(Dataset):
    def __init__(self, data_index):
        # Load features.
        featfure_dir = pathlib.Path("../data/preprocessing/")

        # Target, waypoint
        wp = load_pickle(featfure_dir / "train_waypoint.pkl", verbose=False)
        floor = wp["floor"].to_numpy().astype("int64") + 3
        position = wp[["x", "y"]].to_numpy().astype("float32")
        self.floor = floor[data_index]
        self.position = position[data_index]

        # Build feature.
        site_id = np.load(featfure_dir / "train_site_id.npy")
        self.site_id = site_id[data_index]

        # Wifi features.
        wifi_bssid = np.load(featfure_dir / "train_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "train_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "train_wifi_freq.npy")
        wifi_last_seen_ts = np.load(featfure_dir / "train_wifi_last_seen_ts.npy")

        self.wifi_bssid = wifi_bssid[data_index]
        self.wifi_rssi = wifi_rssi[data_index]
        self.wifi_freq = wifi_freq[data_index]
        self.wifi_last_seen_ts = wifi_freq[data_index]

        # Beacon featurees.
        beacon_uuid = np.load(featfure_dir / "train_beacon_uuid.npy")
        beacon_tx_power = np.load(featfure_dir / "train_beacon_tx_power.npy")
        beacon_rssi = np.load(featfure_dir / "train_beacon_rssi.npy")

        self.beacon_uuid = beacon_uuid[data_index]
        self.beacon_tx_power = beacon_tx_power[data_index]
        self.beacon_rssi = beacon_rssi[data_index]

        # Acce features.
        feature_name = "acce"
        acce_past_x = np.load(featfure_dir / f"train_{feature_name}_past_X.npy")
        acce_past_y = np.load(featfure_dir / f"train_{feature_name}_past_Y.npy")
        acce_past_z = np.load(featfure_dir / f"train_{feature_name}_past_Z.npy")

        acce_feat_x = np.load(featfure_dir / f"train_{feature_name}_feat_X.npy")
        acce_feat_y = np.load(featfure_dir / f"train_{feature_name}_feat_Y.npy")
        acce_feat_z = np.load(featfure_dir / f"train_{feature_name}_feat_Z.npy")

        self.acce_past_x = acce_past_x[data_index]
        self.acce_past_y = acce_past_y[data_index]
        self.acce_past_z = acce_past_z[data_index]
        self.acce_feat_x = acce_feat_x[data_index]
        self.acce_feat_y = acce_feat_y[data_index]
        self.acce_feat_z = acce_feat_z[data_index]

        # Gyroscope features.
        feature_name = "gyroscope"
        gyro_past_x = np.load(featfure_dir / f"train_{feature_name}_past_X.npy")
        gyro_past_y = np.load(featfure_dir / f"train_{feature_name}_past_Y.npy")
        gyro_past_z = np.load(featfure_dir / f"train_{feature_name}_past_Z.npy")

        gyro_feat_x = np.load(featfure_dir / f"train_{feature_name}_feat_X.npy")
        gyro_feat_y = np.load(featfure_dir / f"train_{feature_name}_feat_Y.npy")
        gyro_feat_z = np.load(featfure_dir / f"train_{feature_name}_feat_Z.npy")

        self.gyro_past_x = gyro_past_x[data_index]
        self.gyro_past_y = gyro_past_y[data_index]
        self.gyro_past_z = gyro_past_z[data_index]
        self.gyro_feat_x = gyro_feat_x[data_index]
        self.gyro_feat_y = gyro_feat_y[data_index]
        self.gyro_feat_z = gyro_feat_z[data_index]

        # Magnetic_field features.
        feature_name = "magnetic_field"
        magnetic_past_x = np.load(featfure_dir / f"train_{feature_name}_past_X.npy")
        magnetic_past_y = np.load(featfure_dir / f"train_{feature_name}_past_Y.npy")
        magnetic_past_z = np.load(featfure_dir / f"train_{feature_name}_past_Z.npy")

        magnetic_feat_x = np.load(featfure_dir / f"train_{feature_name}_feat_X.npy")
        magnetic_feat_y = np.load(featfure_dir / f"train_{feature_name}_feat_Y.npy")
        magnetic_feat_z = np.load(featfure_dir / f"train_{feature_name}_feat_Z.npy")

        self.magnetic_past_x = magnetic_past_x[data_index]
        self.magnetic_past_y = magnetic_past_y[data_index]
        self.magnetic_past_z = magnetic_past_z[data_index]
        self.magnetic_feat_x = magnetic_feat_x[data_index]
        self.magnetic_feat_y = magnetic_feat_y[data_index]
        self.magnetic_feat_z = magnetic_feat_z[data_index]

        # Rotation_vector features.
        feature_name = "rotation_vector"
        rotation_past_x = np.load(featfure_dir / f"train_{feature_name}_past_X.npy")
        rotation_past_y = np.load(featfure_dir / f"train_{feature_name}_past_Y.npy")
        rotation_past_z = np.load(featfure_dir / f"train_{feature_name}_past_Z.npy")

        rotation_feat_x = np.load(featfure_dir / f"train_{feature_name}_feat_X.npy")
        rotation_feat_y = np.load(featfure_dir / f"train_{feature_name}_feat_Y.npy")
        rotation_feat_z = np.load(featfure_dir / f"train_{feature_name}_feat_Z.npy")

        self.rotation_past_x = rotation_past_x[data_index]
        self.rotation_past_y = rotation_past_y[data_index]
        self.rotation_past_z = rotation_past_z[data_index]
        self.rotation_feat_x = rotation_feat_x[data_index]
        self.rotation_feat_y = rotation_feat_y[data_index]
        self.rotation_feat_z = rotation_feat_z[data_index]

    def __len__(self):
        return len(self.site_id)

    def __getitem__(self, idx):
        x_build = (self.site_id[idx], self.floor[idx])
        x_wifi = (
            self.site_id[idx],
            self.floor[idx],
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
        x_acce = (
            self.acce_past_x[idx],
            self.acce_past_y[idx],
            self.acce_past_z[idx],
            self.acce_feat_x[idx],
            self.acce_feat_y[idx],
            self.acce_feat_z[idx],
        )
        x_gyroscope = (
            self.gyro_past_x[idx],
            self.gyro_past_y[idx],
            self.gyro_past_z[idx],
            self.gyro_feat_x[idx],
            self.gyro_feat_y[idx],
            self.gyro_feat_z[idx],
        )
        x_magnetic_feild = (
            self.magnetic_past_x[idx],
            self.magnetic_past_y[idx],
            self.magnetic_past_z[idx],
            self.magnetic_feat_x[idx],
            self.magnetic_feat_y[idx],
            self.magnetic_feat_z[idx],
        )
        x_rotation_vector = (
            self.rotation_past_x[idx],
            self.rotation_past_y[idx],
            self.rotation_past_z[idx],
            self.rotation_feat_x[idx],
            self.rotation_feat_y[idx],
            self.rotation_feat_z[idx],
        )

        x = (
            x_build,
            x_wifi,
            x_beacon,
            x_acce,
            x_gyroscope,
            x_magnetic_feild,
            x_rotation_vector,
        )
        y = (self.floor[idx], self.position[idx])
        return x, y


class IndoorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_idx, valid_idx, test_idx):
        super().__init__()
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

    def setup(self, stage=None):
        self.train_dataset = IndoorDataset(self.train_idx)
        self.valid_dataset = IndoorDataset(self.valid_idx)
        self.test_dataset = IndoorDataset(self.test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
