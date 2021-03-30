import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from utils import timer

import config


class IndoorDataset(Dataset):
    def __init__(self, waypoint, wifi):
        self.waypoint = waypoint
        self.wifi = wifi

    def __len__(self):
        return len(self.waypoint)

    def __getitem__(self, idx):
        _waypint = self.waypoint[idx]
        _wifi = self.wifi[idx]
        return (
            _waypint,
            _wifi,
        )


def get_dataloader(waypoint, wifi, n_fold):
    train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
    valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")

    train_dataset = IndoorDataset(waypoint[train_idx], wifi[train_idx])
    valid_dataset = IndoorDataset(waypoint[valid_idx], wifi[valid_idx])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader


def main():
    waypoint = np.load("../data/working/train_waypoint.npy")
    wifi = np.load("../data/working/encoded_train_wifi_features.npy")

    for n_fold in range(config.NUM_FOLD):
        train_dataloader, valid_dataloader = get_dataloader(waypoint, wifi, n_fold)

        batch = iter(train_dataloader).__next__()
        print(batch)
        break


if __name__ == "__main__":
    with timer("Train"):
        main()
