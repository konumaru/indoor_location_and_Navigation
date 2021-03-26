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


class InddorMudule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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
    # TODO:
    # Dataset の定義,
    # CAUTION: wifi の bssid を label encoding しないといけない
    # Dataloader で取得できることを確認
    # Model の定義
    # Model の loss が返ってくることを確認
    # 1 fold で学習、評価

    waypoint = np.load("../data/working/train_waypoint.npy")
    wifi = np.load("../data/working/train_wifi_features.npy")

    for n_fold in range(config.NUM_FOLD):
        train_dataloader, valid_dataloader = get_dataloader(waypoint, wifi, n_fold)
        break


if __name__ == "__main__":
    with timer("Train"):
        main()
