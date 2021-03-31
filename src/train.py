import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

import pytorch_lightning as pl

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
        _waypint = self.waypoint[idx, 1:]
        _build = self.build[idx]
        # wifi
        bssid = self.wifi[idx, 0]
        rssi = self.wifi[idx, 1].astype("float32")
        return (_build, bssid, rssi), _waypint


class IndoorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data, test_data):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data

    def setup(self, stage=None):
        train_dataset = IndoorDataset(*self.train_data)
        train_dataset_len = len(train_dataset)
        train_size = int(train_dataset_len * 0.8)
        valid_size = int(train_dataset_len) - train_size

        self.train_dataset, self.valid_dataset = random_split(
            train_dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.test_dataset = IndoorDataset(*self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def get_datamodule(n_fold):
    waypoint = load_pickle("../data/preprocessing/train_target.pkl")
    build = load_pickle("../data/preprocessing/train_build.pkl")
    wifi = load_pickle("../data/preprocessing/train_wifi.pkl")

    train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
    valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")

    train_data = (waypoint[train_idx], build[train_idx], wifi[train_idx])
    test_data = (waypoint[valid_idx], build[valid_idx], wifi[valid_idx])

    print(*train_data)

    data_module = IndoorDataModule(config.BATCH_SIZE, train_data, test_data)
    return data_module


def main():
    for n_fold in range(config.NUM_FOLD):
        data_module = get_datamodule(n_fold)
        model = InddorModel()

        trainer = pl.Trainer()
        trainer.fit(model=model, datamodule=data_module)
        break


if __name__ == "__main__":
    with timer("Train"):
        main()
