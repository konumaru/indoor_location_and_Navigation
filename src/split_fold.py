import pathlib
import numpy as np
from sklearn.model_selection import GroupKFold

from utils import timer


def main():
    # TODO:

    waypoint = np.load("../data/working/train_waypoint.npy")
    wifi = np.load("../data/working/train_wifi_features.npy")

    path = waypoint[:, 2]
    cv = GroupKFold(n_splits=5)
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(waypoint, groups=path)):
        print(
            f"Fold {n_fold:>02} Train size: {len(train_idx)} Valid size: {len(valid_idx)}"
        )

        np.save(f"../data/fold/fold{n_fold:>02}_waypoint.npy", waypoint[valid_idx])
        np.save(f"../data/fold/fold{n_fold:>02}_wifi.npy", wifi[valid_idx])


if __name__ == "__main__":
    with timer("Split Data"):
        main()
