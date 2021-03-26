import numpy as np
from torch.utils.data import Dataset, DataLoader


class IndoorDataset(Dataset):
    def __init__(self, waypoint_path, wifi_path):
        self.waypoint = np.load(waypoint_path)
        self.wifi = np.load(wifi_path)

    def __len__(self):
        return len(self.waypoint)

    def __getitem__(self, idx):
        _waypint = self.waypoint[idx]
        _wifi = self.wifi[idx]
        return (
            _waypint,
            _wifi,
        )


def main():
    # TODO:
    # Dataset の定義,
    # CAUTION: wifi の bssid を label encoding しないといけない
    # Dataloader で取得できることを確認
    # Model の定義
    # Model の loss が返ってくることを確認
    # 1 fold で学習、評価

    dataset = IndoorDataset(
        "../data/fold/fold00_waypoint.npy", "../data/fold/fold00_wifi.npy"
    )
    print(dataset[1])


if __name__ == "__main__":
    main()
