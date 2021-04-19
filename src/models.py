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
        diff_f = torch.abs(y_hat[:, 0] - y[:, 0])
        diff_x = y_hat[:, 1] - y[:, 1]
        diff_y = y_hat[:, 2] - y[:, 2]

        error = torch.sqrt(diff_x ** 2 + diff_y ** 2) + p * diff_f
        return torch.mean(error)


class BuildModel(nn.Module):
    def __init__(
        self,
        site_embed_dim: int = 64,
    ):
        super(BuildModel, self).__init__()
        self.site_embed_dim = site_embed_dim
        self.output_dim = site_embed_dim
        self.embed_site = nn.Embedding(205, site_embed_dim)

    def forward(self, x):
        site = x[0]

        x = self.embed_site(x)
        x = x.view(-1, self.site_embed_dim)
        return x


class WifiModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 20,
        bssid_embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super(WifiModel, self).__init__()
        self.seq_len = seq_len
        self.bssid_embed_dim = bssid_embed_dim
        self.output_dim = output_dim
        self.embed_bssid = nn.Embedding(237452 + 1, bssid_embed_dim)

        # LSTM layers.
        n_dim_lstm = bssid_embed_dim + 2
        self.lstm_out_dim = 16
        self.lstm1 = nn.LSTM(n_dim_lstm, 128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, self.lstm_out_dim, batch_first=True)

        self.layers = nn.Sequential(
            nn.BatchNorm1d(seq_len * self.lstm_out_dim),
            nn.Linear(seq_len * self.lstm_out_dim, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim),
            nn.PReLU(),
        )

    def forward(self, x):
        (wifi_bssid, wifi_rssi, wifi_freq) = x

        bssid_vec = self.embed_bssid(wifi_bssid)
        wifi_rssi = wifi_rssi.view(-1, self.seq_len, 1)
        wifi_freq = wifi_freq.view(-1, self.seq_len, 1)
        x = torch.cat((bssid_vec, wifi_rssi, wifi_freq), dim=2)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(-1, self.seq_len * self.lstm_out_dim)

        x = self.layers(x)
        return x


class BeaconModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 20,
        uuid_embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super(BeaconModel, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.embed_uuid = nn.Embedding(662, uuid_embed_dim)
        # LSTM layers.
        n_dim_lstm = uuid_embed_dim + 2
        self.lstm_out_dim = 16
        self.lstm1 = nn.LSTM(n_dim_lstm, 128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, self.lstm_out_dim, batch_first=True)

        self.layers = nn.Sequential(
            nn.BatchNorm1d(seq_len * self.lstm_out_dim),
            nn.Linear(seq_len * self.lstm_out_dim, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim),
            nn.PReLU(),
        )

    def forward(self, x):
        uuid, tx_power, rssi = x

        x_uuid = self.embed_uuid(uuid)
        tx_power = tx_power.view(-1, self.seq_len, 1)
        rssi = rssi.view(-1, self.seq_len, 1)

        x = torch.cat((x_uuid, tx_power, rssi), dim=2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(-1, self.seq_len * self.lstm_out_dim)
        x = self.layers(x)
        return x


class InddorModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super(InddorModel, self).__init__()
        self.lr = lr

        self.loss_fn = MeanPositionLoss()

        self.model_build = BuildModel()
        self.model_wifi = WifiModel()
        self.model_beacon = BeaconModel()

        input_dim = (
            self.model_build.output_dim
            + self.model_wifi.output_dim
            + self.model_beacon.output_dim
        )

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.layer_floor = nn.Sequential(
            nn.Linear(64, 14),
            nn.Softmax(dim=1),
        )
        self.layer_position = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x_build, x_wifi, x_beacon = x
        x_build = self.model_build(x_build)
        x_wifi = self.model_wifi(x_wifi)
        x_beacon = self.model_beacon(x_beacon)

        x = torch.cat((x_build, x_wifi, x_beacon), dim=1)
        x = self.layers(x)
        floor = self.layer_floor(x)
        floor = (torch.argmax(floor, axis=1) - 3).view(-1, 1)
        pos = self.layer_position(x)
        x = torch.cat((floor, pos), dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("test_loss", loss)
        return loss
