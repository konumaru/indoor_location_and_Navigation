import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from utils.common import timer
from utils.common import load_pickle

from models import InddorModel, MeanPositionLoss


DEBUG = False

if DEBUG:
    from config import DebugConfig as Config
else:
    from config import Config as Config


class IndoorDataset(Dataset):
    def __init__(self, data_index):
        # Load features.
        featfure_dir = pathlib.Path("../data/preprocessing/")

        # Target, waypoint
        wp = load_pickle(featfure_dir / "train_waypoint.pkl", verbose=False)
        wp = wp[["floor", "x", "y"]].to_numpy()

        self.wp = wp[data_index]

        # Build feature.
        site_id = np.load(featfure_dir / "train_site_id.npy")

        self.site_id = site_id[data_index].reshape(-1, 1)

        # Wifi features.
        wifi_bssid = np.load(featfure_dir / "train_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "train_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "train_wifi_freq.npy")

        self.wifi_bssid = wifi_bssid[data_index][:, :20]
        self.wifi_rssi = wifi_rssi[data_index][:, :20]
        self.wifi_freq = wifi_freq[data_index][:, :20]

        # Beacon featurees.
        beacon_uuid = np.load(featfure_dir / "train_beacon_uuid.npy")
        beacon_tx_power = np.load(featfure_dir / "train_beacon_tx_power.npy")
        beacon_rssi = np.load(featfure_dir / "train_beacon_rssi.npy")

        self.beacon_uuid = beacon_uuid[data_index]
        self.beacon_tx_power = beacon_tx_power[data_index]
        self.beacon_rssi = beacon_rssi[data_index]

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
        y = self.wp[idx]
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
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )


def main():
    pl.seed_everything(Config.SEED)

    for n_fold in range(Config.NUM_FOLD):
        # Load index and select fold daata.
        train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
        valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")
        test_idx = np.load(f"../data/fold/fold{n_fold:>02}_test_idx.npy")

        # Define and setup datamodule.
        datamodule = IndoorDataModule(Config.BATCH_SIZE, train_idx, valid_idx, test_idx)
        datamodule.setup()

        model = InddorModel(lr=1e-2)
        trainer = Trainer(
            accelerator=Config.accelerator,
            gpus=Config.gpus,
            max_epochs=Config.NUM_EPOCH,
            callbacks=Config.callbacks,
            logger=Config.logger,
            fast_dev_run=Config.DEV_RUN,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

        break


if __name__ == "__main__":
    with timer("Train"):
        main()
