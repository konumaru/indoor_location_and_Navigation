import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from utils import timer
from utils import load_pickle

import config


class IndoorDataset(Dataset):
    def __init__(self, waypoint, build, wifi):
        self.waypoint = waypoint
        self.build = build
        self.wifi = wifi

    def __len__(self):
        return len(self.waypoint)

    def __getitem__(self, idx):
        _waypint = self.waypoint[idx]
        _build = self.build[idx]
        # wifi
        bssid = self.wifi[idx, 0]
        rssi = self.wifi[idx, 1].astype("float64")
        return (_waypint, _build, bssid, rssi)


def get_dataloader(n_fold):
    waypoint = load_pickle("../data/preprocessing/train_target.pkl")
    build = load_pickle("../data/preprocessing/train_build.pkl")
    wifi = load_pickle("../data/preprocessing/train_wifi.pkl")

    train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
    valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")

    train_dataset = IndoorDataset(
        waypoint[train_idx], build[train_idx], wifi[train_idx]
    )
    valid_dataset = IndoorDataset(
        waypoint[valid_idx], build[valid_idx], wifi[valid_idx]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader


def main():

    for n_fold in range(config.NUM_FOLD):
        train_dataloader, valid_dataloader = get_dataloader(n_fold)

        print(iter(train_dataloader).next())
        break


if __name__ == "__main__":
    with timer("Train"):
        main()
