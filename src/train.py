import pathlib
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.common import timer
from utils.common import load_pickle

from model import InddorModel, MeanPositionLoss


DEBUG = False

if DEBUG:
    from config import DebugConfig as Config
else:
    from config import Config as Config


class IndoorDataset(Dataset):
    def __init__(self, data_index):
        # Load features.
        featfure_dir = pathlib.Path("../data/preprocessing/")

        site_id = np.load(featfure_dir / "train_site_id.npy")

        wifi_bssid = np.load(featfure_dir / "train_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "train_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "train_wifi_freq.npy")
        wifi_ts_diff = np.load(featfure_dir / "train_wifi_ts_diff.npy")

        wp = load_pickle(featfure_dir / "train_waypoint.pkl", verbose=False)
        wp = wp[["floor", "x", "y"]].to_numpy()

        self.site_id = site_id[data_index].reshape(-1, 1)

        self.wifi_bssid = wifi_bssid[data_index][:, :20]
        self.wifi_rssi = wifi_rssi[data_index][:, :20]
        self.wifi_freq = wifi_freq[data_index][:, :20]
        self.wifi_ts_diff = wifi_ts_diff[data_index][:, :20]

        self.wp = wp[data_index]

    def __len__(self):
        return len(self.site_id)

    def __getitem__(self, idx):
        x_build = self.site_id[idx]
        x_wifi = (
            self.wifi_bssid[idx],
            self.wifi_rssi[idx],
            self.wifi_freq[idx],
            self.wifi_ts_diff[idx],
        )
        x = (x_build, x_wifi)
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

        model = InddorModel(lr=1e-3)
        checkpoint_callback = ModelCheckpoint(monitor="valid_loss")
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            min_delta=0.01,
            patience=20,
            verbose=False,
            mode="min",
        )
        tb_logger = TensorBoardLogger(save_dir="../tb_logs", name="Baseline")

        # dataloader = datamodule.train_dataloader()
        # batch = next(iter(dataloader))
        # x, y = batch

        # model = InddorModel()
        # z = model(x)
        # print(z)
        # loss_fn = MeanPositionLoss()
        # loss = loss_fn(z, y)
        # print(loss)

        trainer = Trainer(
            accelerator=Config.accelerator,
            gpus=Config.gpus,
            max_epochs=Config.NUM_EPOCH,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=tb_logger,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
        break


if __name__ == "__main__":
    with timer("Train"):
        main()
