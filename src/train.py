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

import config


class IndoorDataset(Dataset):
    def __init__(self, data_index):
        # Load features.
        featfure_dir = pathlib.Path("../data/preprocessing/")

        site_id = np.load(featfure_dir / "train_site_id.npy")
        site_height = np.load(featfure_dir / "train_site_height.npy")
        site_width = np.load(featfure_dir / "train_site_width.npy")

        wifi_bssid = np.load(featfure_dir / "train_wifi_bssid.npy")
        wifi_rssi = np.load(featfure_dir / "train_wifi_rssi.npy")
        wifi_freq = np.load(featfure_dir / "train_wifi_freq.npy")

        wp = load_pickle(featfure_dir / "train_waypoint.pkl")
        wp = wp[["floor", "x", "y"]].to_numpy()

        self.site_id = site_id[data_index]
        self.site_height = site_height[data_index]
        self.site_width = site_width[data_index]

        self.wifi_bssid = wifi_bssid[data_index]
        self.wifi_rssi = wifi_rssi[data_index]
        self.wifi_freq = wifi_freq[data_index]

        self.wp = wp[data_index]

    def __len__(self):
        return len(self.site_id)

    def __getitem__(self, idx):
        x = (
            self.site_id[idx],
            self.site_height[idx],
            self.site_width[idx],
            self.wifi_bssid[idx],
            self.wifi_rssi[idx],
            self.wifi_freq[idx],
        )
        y = self.wp[idx]
        return x, y


class IndoorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, train_idx, valid_idx, test_idx, stage=None):
        self.train_dataset = IndoorDataset(train_idx)
        self.valid_dataset = IndoorDataset(valid_idx)
        self.test_dataset = IndoorDataset(test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=False,
        )


def main():
    pl.seed_everything(config.SEED)

    for n_fold in range(config.NUM_FOLD):
        # Load index and select fold daata.
        train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
        valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")
        test_idx = np.load(f"../data/fold/fold{n_fold:>02}_test_idx.npy")

        print(test_idx.max())
        featfure_dir = pathlib.Path("../data/preprocessing/")
        site_id = np.load(featfure_dir / "train_site_id.npy")
        print(site_id.shape)
        # print(site_id[test_idx])

        break

        # Define and setup datamodule.
        datamodule = IndoorDataModule(config.BATCH_SIZE)
        datamodule.setup(train_idx, valid_idx, test_idx)

    #     model = InddorModel(lr=1e-4)
    #     checkpoint_callback = ModelCheckpoint(monitor="valid_loss")
    #     early_stop_callback = EarlyStopping(
    #         monitor="valid_loss",
    #         min_delta=0.00,
    #         patience=20,
    #         verbose=False,
    #         mode="min",
    #     )
    #     tb_logger = TensorBoardLogger(
    #         save_dir="../tb_logs", name="wifiLSTM_buidModel_prod"
    #     )

    #     # dataloader = datamodule.train_dataloader()
    #     # batch = next(iter(dataloader))
    #     # x, y = batch

    #     # model = InddorModel()
    #     # z = model(x)
    #     # print(z)
    #     # loss_fn = MeanPositionLoss()
    #     # loss = loss_fn(z, y)
    #     # print(loss)

    #     trainer = Trainer(
    #         accelerator="dp",
    #         gpus=1,
    #         max_epochs=200,
    #         callbacks=[checkpoint_callback, early_stop_callback],
    #         logger=tb_logger,
    #     )
    #     trainer.fit(model=model, datamodule=datamodule)
    #     trainer.test(model=model, datamodule=datamodule)
    # break


if __name__ == "__main__":
    with timer("Train"):
        main()
