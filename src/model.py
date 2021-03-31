import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class WifiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # bssi
        # Nunique of bssid is 185859.
        self.bssid_embed = nn.Embedding(185859, 64)
        # rssi
        self.bn1 = nn.BatchNorm1d(100)
        self.layer1 = nn.Linear(100)

    def forward(self, x):
        bssid, rssi = x[0], x[1]
        bssid = F.relu(self.bssid_embed(bssid))
        rssi = F.relu(self.layer1(rssi))
        x = torch.cat([bssid, rssi], dim=0)

        return x


class InddorModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()

    def forward(self, x):
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
    trainer = pl.Trainer()
    model = InddorModel()

    # print(torch.rand(size=(1000, 3, 100)))
    # trainer.fit(InddorModel)


if __name__ == "__main__":
    main()
