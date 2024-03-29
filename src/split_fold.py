import pathlib
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    GroupKFold,
    GroupShuffleSplit,
)

from utils.common import timer
from utils.common import load_pickle, dump_pickle

from config import Config as Config


def main():
    # ids, (site, floor, path)
    wp = load_pickle("../data/preprocessing/train_waypoint.pkl")

    le = LabelEncoder()
    wp["path"] = le.fit_transform(wp["path"])

    # NOTE: Conclusion at v1 is below.
    # - Stratifield is bad cv strategy.
    # - k <= 5.
    # - Not Shuffle Split.

    cv = GroupKFold(n_splits=Config.NUM_FOLD)
    for n_fold, (train_idx, test_idx) in enumerate(
        cv.split(X=wp, y=wp[["path"]], groups=wp[["path"]])
    ):
        print(f"\nFold {n_fold:>02}")

        valid_idx, test_idx = train_test_split(test_idx, test_size=0.5)

        train_unique = np.unique(wp.loc[train_idx, "path"])
        valid_unique = np.unique(wp.loc[valid_idx, "path"])
        test_unique = np.unique(wp.loc[test_idx, "path"])

        print(
            f"\tTrain size: {len(train_idx)}",
            f"Valid size: {len(valid_idx)}",
            f"Test size: {len(test_idx)}",
        )
        print(
            f"\tTrain nunique: {len(train_unique)}",
            f"Valid nunique: {len(valid_unique)}",
            f"Test nunique: {len(test_unique)}",
        )

        np.save(f"../data/fold/fold{n_fold:>02}_train_idx.npy", train_idx)
        np.save(f"../data/fold/fold{n_fold:>02}_valid_idx.npy", valid_idx)
        np.save(f"../data/fold/fold{n_fold:>02}_test_idx.npy", test_idx)


if __name__ == "__main__":
    with timer("Split Data"):
        main()
