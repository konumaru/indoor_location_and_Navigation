import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class MeanPositionLoss(nn.Module):
    def __init__(self):
        super(MeanPositionLoss, self).__init__()

    def forward(self, y_hat, y):
        p = 15
        diff_f = y_hat[0] - y[0]
        diff_x = y_hat[1] - y[2]
        diff_y = y_hat[1] - y[2]

        error = torch.sqrt(diff_x * diff_x + diff_y * diff_y) + p * torch.sqrt(
            diff_f * diff_f
        )
        return torch.mean(error)


class WifiModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 100,
        bssid_embed_dim: int = 64,
        output_dim: int = 512,
    ):
        super().__init__()
        input_size = 400 + bssid_embed_dim * seq_len
        self.seq_len = seq_len
        self.bssid_embed_dim = bssid_embed_dim
        self.output_dim = output_dim
        # Nunique of bssid is 185859.
        self.embed_bssid = nn.Embedding(180203 + 1, bssid_embed_dim)
        self.layers = nn.Sequential(
            # Layer 0
            nn.BatchNorm1d(input_size),
            nn.Dropout(0.2),
            # Layer 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            # Layer 2
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            # Layer 3
            nn.Linear(512, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        bssid, rssi, freq, ts_diff, last_seen_ts_diff = x

        bssid = self.embed_bssid(bssid)
        bssid = bssid.view(-1, self.seq_len * self.bssid_embed_dim)
        rssi = rssi.view(-1, self.seq_len)
        freq = freq.view(-1, self.seq_len)
        ts_diff = ts_diff.view(-1, self.seq_len)
        last_seen_ts_diff = last_seen_ts_diff.view(-1, self.seq_len)

        x = torch.cat((bssid, rssi, freq, ts_diff, last_seen_ts_diff), dim=1)
        x = self.layers(x)
        return x


class BuildModel(nn.Module):
    def __init__(
        self,
        site_embed_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.site_embed_dim = site_embed_dim
        self.output_dim = output_dim

        self.embed_site = nn.Embedding(205, site_embed_dim)
        self.layer1 = nn.Linear(site_embed_dim, output_dim)

    def forward(self, x):
        x = self.embed_site(x)
        x = x.view(-1, self.site_embed_dim)
        x = F.relu(self.layer1(x))
        return x


class InddorModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.loss_fn = MeanPositionLoss()
        self.model_wifi = WifiModel()
        self.model_build = BuildModel()

        output_dim = self.model_wifi.output_dim + self.model_build.output_dim

        self.layers = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        input_build, input_wifi = x
        build_feature = self.model_build(input_build)
        wifi_feature = self.model_wifi(input_wifi)

        x = torch.cat((build_feature, wifi_feature), dim=1)
        x = self.layers(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(*z, *y)
        metric = self._comp_metric((y[0], z), y)
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("train_loss", loss)
        self.log("train_metric", metric)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("valid_loss", loss)
        self.log("valid_metric", metric)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_metric", metric)

    def _comp_metric(self, y_hat, y):
        def rmse(y_hat, y):
            return torch.sqrt(F.mse_loss(y_hat, y))

        p = 15
        floor_hat, waypoint_hat = y_hat
        floor, waypoint = y
        metric = torch.mean(
            rmse(waypoint_hat, waypoint) + p * torch.abs(floor_hat - floor)
        )
        return metric


def main():
    pl.seed_everything(42)
    batch_size = 10
    # targets.
    floor = torch.randint(100, size=(batch_size, 1))
    waypoint = torch.rand(size=(batch_size, 2))
    y = torch.cat((floor, waypoint), dim=1)
    # build features.
    build = torch.randint(100, size=(batch_size, 1))
    input_build = build
    # wifi feaatures.
    bssid = torch.randint(100, size=(batch_size, 100))
    rssi = torch.rand(size=(batch_size, 100))
    freq = torch.rand(size=(batch_size, 100))
    ts_dff = torch.rand(size=(batch_size, 100))
    last_seen_ts_dff = torch.rand(size=(batch_size, 100))
    input_wifi = (bssid, rssi, freq, ts_dff, last_seen_ts_dff)

    model = WifiModel()
    z = model(input_wifi)
    print(z)

    x, y = (input_build, input_wifi), y

    model = InddorModel()
    z = model(x)

    loss_fn = MeanPositionLoss()
    loss = loss_fn(z, y)
    print(loss)


if __name__ == "__main__":
    main()
