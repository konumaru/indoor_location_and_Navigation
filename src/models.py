import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_hat, y):
        diff_x = y_hat[:, 0] - y[:, 0]
        diff_y = y_hat[:, 1] - y[:, 1]
        error = torch.sqrt(diff_x ** 2 + diff_y ** 2)
        return torch.mean(error)


class MeanPositionLoss(nn.Module):
    def __init__(self):
        super(MeanPositionLoss, self).__init__()

    def forward(self, y_hat, y):
        floor, pos = y
        floor_hat, pos_hat = y_hat

        p = 1  # 15
        diff_f = torch.abs(floor_hat - floor)
        diff_x = pos_hat[:, 0] - pos[:, 0]
        diff_y = pos_hat[:, 1] - pos[:, 1]

        error = torch.sqrt(torch.pow(diff_x, 2) + torch.pow(diff_y, 2)) + p * diff_f
        return torch.mean(error)


class BuildModel(nn.Module):
    def __init__(
        self,
        site_embed_dim: int = 64,
    ):
        super(BuildModel, self).__init__()
        # TODO: floor and embed_floorを特徴量として追加する
        self.site_embed_dim = site_embed_dim
        self.output_dim = site_embed_dim

        self.embed_site = nn.Embedding(205 + 1, site_embed_dim)

    def forward(self, x):
        site = x[0]

        x = self.embed_site(x)
        x = x.view(-1, self.site_embed_dim)
        return x


class WifiModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 100,
        bssid_embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super(WifiModel, self).__init__()
        self.seq_len = seq_len
        self.bssid_embed_dim = bssid_embed_dim
        self.output_dim = output_dim

        self.embed_bssid = nn.Embedding(238859 + 1, bssid_embed_dim)
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

        self.embed_uuid = nn.Embedding(663 + 1, uuid_embed_dim)
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
        # Define loss function.
        self.loss_fn = MeanPositionLoss()  # RMSELoss()
        # Each data models.
        self.model_build = BuildModel()
        self.model_wifi = WifiModel()
        self.model_beacon = BeaconModel()

        input_dim = (
            self.model_build.output_dim
            + self.model_wifi.output_dim
            + self.model_beacon.output_dim
        )

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.layer_floor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 14),
            nn.Softmax(dim=1),
        )
        self.layer_position = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x_build, x_wifi, x_beacon = x

        x_build = self.model_build(x_build)
        x_wifi = self.model_wifi(x_wifi)
        x_beacon = self.model_beacon(x_beacon)

        x = torch.cat((x_build, x_wifi, x_beacon), dim=1)
        x = self.layers(x)

        f = self.layer_floor(x)
        f = (torch.argmax(f, axis=1) - 3).view(-1, 1)
        pos = self.layer_position(x)
        return (f, pos)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995),
            "name": "lr",
        }
        return [optimizer], [lr_scheduler]

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y)
        return z, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_epoch_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log("valid_loss", loss)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("valid_epoch_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z, loss = self.shared_step(batch)
        self.log("test_loss", loss)

        outputs = {
            "test_loss": loss,
            "floor": y[0].detach().cpu().numpy(),
            "floor_hat": z[0].detach().cpu().numpy(),
            "position": y[1].detach().cpu().numpy(),
            "position_hat": z[1].detach().cpu().numpy(),
        }
        return outputs

    def test_epoch_end(self, test_outputs):
        floor = np.concatenate([output["floor"] for output in test_outputs], axis=0)
        floor_hat = np.concatenate(
            [output["floor_hat"] for output in test_outputs], axis=0
        )
        position = np.concatenate(
            [output["position"] for output in test_outputs], axis=0
        )
        position_hat = np.concatenate(
            [output["position_hat"] for output in test_outputs], axis=0
        )

        # Save plot of floor count.
        # https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#tensorboard
        figure = self.floor_bar_plot(floor, floor_hat)
        self.logger.experiment.add_figure("floor_cnt", figure, 0)
        # Save plot of position distribution.
        self.logger.experiment.add_histogram("x", position[:, 0], 0)
        self.logger.experiment.add_histogram("x_hat", position_hat[:, 0], 0)
        self.logger.experiment.add_histogram("y", position[:, 1], 0)
        self.logger.experiment.add_histogram("y_hat", position_hat[:, 1], 0)

    def floor_bar_plot(self, floor, floor_hat):
        idx_all = np.arange(14) - 3
        idx, cnt = np.unique(floor, return_counts=True)
        idx_hat, cnt_hat = np.unique(floor_hat, return_counts=True)

        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(idx - width / 2, cnt, width, label="Predict")
        rects2 = ax.bar(idx_hat + width / 2, cnt_hat, width, label="Actual")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Record Count")
        ax.set_title("Floor Count")
        ax.set_xticks(idx_all)
        ax.set_xticklabels(idx_all)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        return fig
