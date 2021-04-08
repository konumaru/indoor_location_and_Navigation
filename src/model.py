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
        seq_len: int = 20,
        bssid_embed_dim: int = 16,
        output_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.bssid_embed_dim = bssid_embed_dim
        self.output_dim = output_dim
        # Nunique of bssid is ___.
        self.embed_bssid = nn.Embedding(180203 + 1, bssid_embed_dim)

        site_embed_dim = 16
        self.layers_build = nn.Sequential(
            nn.BatchNorm1d(site_embed_dim),
            nn.Linear(site_embed_dim, seq_len),
            nn.ReLU(),
        )
        # LSTM layers.
        self.bn1 = nn.BatchNorm1d(seq_len)
        self.lstm1 = nn.LSTM(18, 128)
        self.lstm2 = nn.LSTM(128, 64)

        self.layers = nn.Sequential(
            nn.BatchNorm1d(seq_len * 64),
            nn.Linear(seq_len * 64, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        ouput_build, (bssid, rssi, freq, ts_diff, last_seen_ts_diff) = x

        build = self.layers_build(ouput_build)
        build = build.view(-1, self.seq_len, 1)

        bssid = self.embed_bssid(bssid)
        feat_location = torch.cat((build, bssid), dim=2)

        x = torch.cat((feat_location, rssi.view(-1, self.seq_len, 1)), dim=2)
        # x = torch.cat((x, freq.view(-1, 100, 1)), dim=2)
        # x = torch.cat((x, ts_diff.view(-1, 100, 1)), dim=2)
        # x = torch.cat((x, last_seen_ts_diff.view(-1, 100, 1)), dim=2)

        x = self.bn1(x)
        x = nn.Dropout(0.2)(x)
        x = x.transpose(0, 1)
        x, _ = self.lstm1(x)
        x = nn.Dropout(0.3)(x)
        x = F.relu(x)
        x, _ = self.lstm2(x)
        x = nn.Dropout(0.1)(x)
        x = F.relu(x)

        x = x.transpose(0, 1)
        x = x.reshape(-1, self.seq_len * 64)

        x = self.layers(x)
        return x


class BuildModel(nn.Module):
    def __init__(
        self,
        site_embed_dim: int = 16,
        output_dim: int = 16,
    ):
        super().__init__()
        self.site_embed_dim = site_embed_dim
        self.output_dim = output_dim
        self.embed_site = nn.Embedding(205, site_embed_dim)

    def forward(self, x):
        x = self.embed_site(x)
        x = x.view(-1, self.site_embed_dim)
        return x


class InddorModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.loss_fn = MeanPositionLoss()
        self.model_wifi = WifiModel()
        self.model_build = BuildModel()

        output_dim = self.model_wifi.output_dim

        self.layers = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.layer_floor = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 13),
            nn.Softmax(dim=1),
        )
        self.layer_position = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        input_build, input_wifi = x
        build_feature = self.model_build(input_build)
        wifi_feature = self.model_wifi((build_feature, input_wifi))

        x = self.layers(wifi_feature)
        floor = self.layer_floor(x)
        floor = (torch.argmax(floor, dim=1) - 3).view(-1, 1)
        pos = self.layer_position(x)
        x = torch.cat((floor, pos), dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        metric = self._comp_metric(z, y)
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("test_loss", loss)
        return loss

    def _comp_metric(self, y_hat, y):
        p = 15
        diff_f = y_hat[0] - y[0]
        diff_x = y_hat[1] - y[2]
        diff_y = y_hat[1] - y[2]

        error = torch.sqrt(diff_x * diff_x + diff_y * diff_y) + p * torch.sqrt(
            diff_f * diff_f
        )
        return torch.mean(error)


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
    seq_len = 20
    bssid = torch.randint(100, size=(batch_size, seq_len))
    rssi = torch.rand(size=(batch_size, seq_len))
    freq = torch.rand(size=(batch_size, seq_len))
    ts_dff = torch.rand(size=(batch_size, seq_len))
    last_seen_ts_dff = torch.rand(size=(batch_size, seq_len))
    input_wifi = (bssid, rssi, freq, ts_dff, last_seen_ts_dff)

    # Test BuildModel
    model_build = BuildModel()
    output_build = model_build(input_build)
    print(output_build.shape)

    model = WifiModel()
    output_wifi = model((output_build, input_wifi))
    print(output_wifi.shape)

    x, y = (input_build, input_wifi), y

    model = InddorModel()
    z = model(x)
    print(z)

    loss_fn = MeanPositionLoss()
    loss = loss_fn(z, y)
    print(loss)


if __name__ == "__main__":
    main()
