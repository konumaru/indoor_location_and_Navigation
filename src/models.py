import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class MeanAbsolutePositionLoss(nn.Module):
    def __init__(self):
        super(MeanAbsolutePositionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_hat, y, floor_hat, floor):
        pos_error = torch.abs(y_hat - y)
        pos_error = torch.sum(pos_error, dim=1)

        p = 1  # 15
        floor_error = p * self.ce_loss(floor_hat, floor)

        error = pos_error + floor_error
        return torch.mean(error)


class MeanPositionLoss(nn.Module):
    def __init__(self):
        super(MeanPositionLoss, self).__init__()

    def forward(self, y_hat, y, floor_hat=None, floor=None):
        pos_error = y_hat - y
        pos_error = torch.sum(torch.sqrt(torch.pow(pos_error, 2)), dim=1)

        p = 15
        floor_error = p * torch.abs(floor_hat - floor)

        error = pos_error + floor_error
        return torch.mean(error)


class BuildModel(nn.Module):
    def __init__(
        self,
        site_embed_dim: int = 64,
    ):
        super(BuildModel, self).__init__()
        # TODO: floor and embed_floorを特徴量として追加する
        self.site_embed_dim = site_embed_dim
        self.output_dim = site_embed_dim + 1

        self.embed_site = nn.Embedding(205 + 1, site_embed_dim)

    def forward(self, x):
        x_site, x_floor = x
        x_site = self.embed_site(x_site)
        x_site = x_site.view(-1, self.site_embed_dim)
        x = torch.cat((x_site, x_floor), dim=1)
        return x


class WifiModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 100,
        site_embed_dim: int = 8,
        bssid_embed_dim: int = 64,
        output_dim: int = 256,
    ):
        super(WifiModel, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.site_embed = nn.Embedding(205, site_embed_dim)
        self.bssid_embed = nn.Embedding(238859 + 1, bssid_embed_dim)
        # LSTM layers.
        n_dim_lstm = bssid_embed_dim + 1
        self.lstm_out_dim = 128
        self.lstm1 = nn.LSTM(n_dim_lstm, 256, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(256, self.lstm_out_dim, batch_first=True)

        layer_input_dim = seq_len * self.lstm_out_dim + site_embed_dim + 1
        self.layers = nn.Sequential(
            nn.BatchNorm1d(layer_input_dim),
            nn.Linear(layer_input_dim, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim),
            nn.PReLU(),
        )

    def forward(self, x):
        (site, floor, wifi_bssid, wifi_rssi, wifi_freq, wifi_last_seen_ts) = x

        wifi_bssid_vec = self.bssid_embed(wifi_bssid)
        wifi_rssi = wifi_rssi.view(-1, self.seq_len, 1)
        # wifi_freq = wifi_freq.view(-1, self.seq_len, 1)
        # wifi_last_seen_ts = wifi_last_seen_ts.view(-1, self.seq_len, 1)
        # x = torch.cat((bssid_vec, wifi_rssi, wifi_freq, wifi_last_seen_ts), dim=2)
        x = torch.cat((wifi_bssid_vec, wifi_rssi), dim=2)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(-1, self.seq_len * self.lstm_out_dim)

        site_vec = self.site_embed(site)
        floor = floor.view(-1, 1)
        x = torch.cat((x, site_vec, floor), dim=1)
        x = self.layers(x)
        return x


class BeaconModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 20,
        uuid_embed_dim: int = 64,
        output_dim: int = 128,
    ):
        super(BeaconModel, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.embed_uuid = nn.Embedding(663 + 1, uuid_embed_dim)
        # LSTM layers.
        n_dim_lstm = uuid_embed_dim + 2
        self.lstm_out_dim = 16
        self.lstm1 = nn.LSTM(n_dim_lstm, 256, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(256, self.lstm_out_dim, batch_first=True)

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
    def __init__(self, lr: float = 1e-3, wifi_seq_len: int = 100):
        super(InddorModel, self).__init__()
        self.lr = lr
        # Define loss function.
        self.loss_fn = MeanAbsolutePositionLoss()
        self.eval_fn = MeanPositionLoss()
        # Each data models.
        # self.model_build = BuildModel()
        self.model_wifi = WifiModel(seq_len=wifi_seq_len)
        # self.model_beacon = BeaconModel()

        input_dim = (
            self.model_wifi.output_dim
            # + self.model_beacon.output_dim
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
            nn.ReLU(),
            nn.Linear(64, 14),
        )
        self.layer_position = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x_build, x_wifi, x_beacon = x

        # x_build = self.model_build(x_build)
        x_wifi = self.model_wifi(x_wifi)
        # x_beacon = self.model_beacon(x_beacon)

        x = torch.cat(
            (
                # x_build,
                x_wifi,
                # x_beacon,
            ),
            dim=1,
        )
        x = self.layers(x)

        f = self.layer_floor(x)
        pos = self.layer_position(x)
        return (f, pos)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            "name": "lr",
        }
        return [optimizer], [lr_scheduler]

    def shared_step(self, batch, step_name):
        x, y = batch
        floor_hat, pos_hat = self(x)
        loss = self.loss_fn(pos_hat, y[1], floor_hat, y[0])
        outputs = {"y": y, "z": (torch.argmax(floor_hat, dim=1), pos_hat)}
        self.log(f"{step_name}_loss", loss)
        return loss, outputs

    def shared_epoch_end(self, outputs, name):
        floor = torch.cat([out["outputs"]["y"][0] for out in outputs], dim=0)
        floor_hat = torch.cat([out["outputs"]["z"][0] for out in outputs], dim=0)
        pos = torch.cat([out["outputs"]["y"][1] for out in outputs], dim=0)
        pos_hat = torch.cat([out["outputs"]["z"][1] for out in outputs], dim=0)

        metric = self.eval_fn(pos_hat, pos, floor_hat, floor)
        self.log(f"{name}_metric", metric, prog_bar=True)
        return floor, pos, floor_hat, pos_hat

    def training_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch, "train")
        return {"loss": loss, "outputs": outputs}

    def training_step_end(self, outputs):
        return outputs

    def training_epoch_end(self, outputs):
        _ = self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch, "valid")
        return {"valid_loss": loss, "outputs": outputs}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        _ = self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch, "test")
        return {"test_loss": loss, "outputs": outputs}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        floor, pos, floor_hat, pos_hat = self.shared_epoch_end(outputs, "test")

        floor = floor.detach().cpu().numpy()
        floor_hat = floor_hat.detach().cpu().numpy()
        pos = pos.detach().cpu().numpy()
        pos_hat = pos_hat.detach().cpu().numpy()

        # Save plot of floor count.
        # https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#tensorboard
        figure = self.floor_bar_plot(floor, floor_hat)
        self.logger.experiment.add_figure("floor_cnt", figure, 0)
        # Save plot of position distribution.
        self.logger.experiment.add_histogram("x", pos[:, 0], 0)
        self.logger.experiment.add_histogram("x_hat", pos_hat[:, 0], 0)
        self.logger.experiment.add_histogram("y", pos[:, 1], 0)
        self.logger.experiment.add_histogram("y_hat", pos_hat[:, 1], 0)

    def floor_bar_plot(self, floor, floor_hat):
        idx_all = np.arange(14)
        idx, cnt = np.unique(floor, return_counts=True)
        idx_hat, cnt_hat = np.unique(floor_hat, return_counts=True)

        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(idx - width / 2, cnt, width, label="Actual")
        rects2 = ax.bar(idx_hat + width / 2, cnt_hat, width, label="Predict")

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
