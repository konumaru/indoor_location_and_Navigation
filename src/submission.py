import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from model import InddorModel


class IndoorTestDataset(Dataset):
    def __init__(self, build, wifi):
        self.build = build
        self.wifi = wifi

    def __len__(self):
        return len(self.build)

    def __getitem__(self, idx):
        input_build = self.build[idx]
        # wifi
        input_wifi = (
            self.wifi[idx, 0],  # bssid
            self.wifi[idx, 1].astype("float32"),  # rssi
            self.wifi[idx, 2].astype("float32"),  # frequencyt
            self.wifi[idx, 3].astype("float32"),  # ts_diff
            self.wifi[idx, 4].astype("float32"),  # last_seen_ts_diff
        )
        return (input_build, input_wifi)


from train import InddorModel
from utils import load_pickle


def get_dataloader():
    build = load_pickle("../data/preprocessing/test_build.pkl")
    wifi = load_pickle("../data/preprocessing/test_wifi.pkl")

    dataset = IndoorTestDataset(build, wifi)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=8, drop_last=False)
    return dataloader


def main():
    dataloader = get_dataloader()

    model = InddorModel.load_from_checkpoint(
        "../tb_logs/wifiLSTM_buidModel_prod/version_1/checkpoints/epoch=14-step=5024.ckpt"
    )
    model.eval()
    model.freeze()

    preds = []
    for batch in dataloader:
        y_hat = model(batch)
        preds.append(y_hat)

    pred = torch.cat(preds, dim=0)
    pred = pred.detach().numpy().copy()

    submission = pd.read_csv("../data/raw/sample_submission.csv")
    submission.iloc[:, 1:] = pred

    submission["floor"] = submission["floor"].astype(int)
    submission.to_csv("../data/submit/submission.csv", index=False)


if __name__ == "__main__":
    main()
