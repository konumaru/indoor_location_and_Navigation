import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class WifiModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 100,
        bssid_embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.bssid_embed_dim = bssid_embed_dim
        self.output_dim = output_dim
        # For  bssi
        # Nunique of bssid is 185859.
        self.embed_bssid = nn.Embedding(185860, bssid_embed_dim)
        # For rssi
        self.bn1 = nn.BatchNorm1d(seq_len)
        # LSTM layers
        self.lstm1 = nn.LSTM(bssid_embed_dim + 1, 128)
        self.lstm2 = nn.LSTM(128, 64)
        self.layer1 = nn.Linear(seq_len * 64, 128)
        self.layer2 = nn.Linear(128, output_dim)

    def forward(self, x):
        bssid, rssi = x
        x_bssid = self.embed_bssid(bssid)

        x_rssi = self.bn1(rssi)
        x_rssi = x_rssi.view(-1, self.seq_len, 1)

        x = torch.cat((x_bssid, x_rssi), dim=2)
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)

        x = x.view(-1, self.seq_len * 64)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
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
        self.model_wifi = WifiModel()
        self.model_build = BuildModel()

        output_dim = self.model_wifi.output_dim + self.model_build.output_dim
        self.layer1 = nn.Linear(output_dim, 64)
        self.layer2 = nn.Linear(64, 2)

    def forward(self, x):
        x_build = self.model_build(x[0])
        x_wifi = self.model_wifi((x[1], x[2]))

        x = torch.cat((x_build, x_wifi), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = F.mse_loss(z, y)
        return z, loss

    def training_step(self, batch, batch_idx):
        z, loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        z, loss = self.shared_step(batch)
        self.log("valid_loss", loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, batch, batch_idx):
        z, loss = self.shared_step(batch)

    def _comp_metric(self, y_hat, y):
        def rmse(y_hat, y):
            return torch.sqrt(F.mse_loss(y_hat, y))

        p = 15
        metric = torch.mean(
            rmse(y_hat[:, 1:], y[:, 1:]) + p * torch.abs(y_hat[:, 0] - y[:, 0])
        )
        return metric


def main():
    batch_size = 10
    waypoint = torch.rand(size=(batch_size, 3))
    build = torch.randint(100, size=(batch_size, 1))
    bssid = torch.randint(100, size=(batch_size, 100))
    rssi = torch.rand(size=(batch_size, 100))

    x, y = (build, bssid, rssi), waypoint

    model = InddorModel()
    z = model(x)
    print(z.shape)


if __name__ == "__main__":
    main()
