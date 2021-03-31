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
        self.embed_bssid = nn.Embedding(185859, bssid_embed_dim)
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

        self.embed_site = nn.Embedding(204, site_embed_dim)
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
        pass

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # z = self(x) or self.module(x)
        # loss = loss_func(z, y)
        # return {'loss': loss, 'preds': z}
        pass

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, batch, batch_idx):
        # NOTE:
        """
        # call after training
        trainer = Trainer()
        trainer.fit(model)

        # automatically auto-loads the best weights
        trainer.test(test_dataloaders=test_dataloader)

        # or call with pretrained model
        model = MyLightningModule.load_from_checkpoint(PATH)
        trainer = Trainer()
        trainer.test(model, test_dataloaders=test_dataloader)
        """
        pass


def main():
    batch_size = 10
    waypoint = torch.rand(size=(batch_size, 3))
    build = torch.randint(100, size=(batch_size, 1))
    bssid = torch.randint(100, size=(batch_size, 100))
    rssi = torch.rand(size=(batch_size, 100))

    x, y = (build, bssid, rssi), waypoint

    # model = WifiModel()
    # z = model((bssid, rssi))
    # print(z.shape)

    # model = BuildModel()
    # z = model(build)
    # print(z.shape)

    model = InddorModel()
    z = model(x)
    print(z.shape)

    # trainer = pl.Trainer()
    # model = InddorModel()

    # print(torch.rand(size=(1000, 3, 100)))
    # trainer.fit(InddorModel)


if __name__ == "__main__":
    main()
