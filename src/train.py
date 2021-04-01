import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import timer
from utils import load_pickle

from model import InddorModel

import config


class IndoorDataset(Dataset):
    def __init__(self, waypoint, build, wifi):
        self.waypoint = waypoint
        self.build = build
        self.wifi = wifi

    def __len__(self):
        return len(self.waypoint)

    def __getitem__(self, idx):
        floor = self.waypoint[idx, 0].astype("int16")
        _waypint = self.waypoint[idx, 1:].astype("float32")
        _build = self.build[idx]
        # wifi
        bssid = self.wifi[idx, 0]
        rssi = self.wifi[idx, 1].astype("float32")
        return (_build, bssid, rssi), (floor, _waypint)


class IndoorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data, valid_data, test_data):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

    def setup(self, stage=None):
        self.train_dataset = IndoorDataset(*self.train_data)
        self.valid_dataset = IndoorDataset(*self.valid_data)
        self.test_dataset = IndoorDataset(*self.test_data)

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


def get_datamodule(n_fold):
    waypoint = load_pickle("../data/preprocessing/train_target.pkl")
    build = load_pickle("../data/preprocessing/train_build.pkl")
    wifi = load_pickle("../data/preprocessing/train_wifi.pkl")

    train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
    valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")
    test_idx = np.load(f"../data/fold/fold{n_fold:>02}_test_idx.npy")

    train_data = (waypoint[train_idx], build[train_idx], wifi[train_idx])
    valid_data = (waypoint[valid_idx], build[valid_idx], wifi[valid_idx])
    test_data = (waypoint[test_idx], build[test_idx], wifi[test_idx])

    data_module = IndoorDataModule(config.BATCH_SIZE, train_data, valid_data, test_data)
    data_module.setup()
    return data_module


def main():
    pl.seed_everything(config.SEED)
    for n_fold in range(config.NUM_FOLD):
        datamodule = get_datamodule(n_fold)
        model = InddorModel()
        checkpoint_callback = ModelCheckpoint(monitor="valid_loss")
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="min",
        )
        tb_logger = TensorBoardLogger(save_dir="../tb_logs", name="wifiLSTM_buidModel")

        trainer = Trainer(
            accelerator=None,
            max_epochs=1,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=tb_logger,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

        break


if __name__ == "__main__":
    with timer("Train"):
        main()
