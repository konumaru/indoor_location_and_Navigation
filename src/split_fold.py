import pathlib
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from utils import timer
from utils import load_pickle, dump_pickle

import config


def main():
    # ids, (site, floor, path)
    ids = load_pickle("../data/preprocessing/train_ids.pkl")

    path = ids[:, 2]
    cv = GroupKFold(n_splits=3)
    for n_fold, (train_idx, test_idx) in enumerate(cv.split(ids, groups=path)):
        print(f"Fold {n_fold:>02}")

        valid_idx, test_idx = train_test_split(test_idx, test_size=0.5)

        train_unique = np.unique(ids[train_idx, 2])
        valid_unique = np.unique(ids[valid_idx, 2])
        test_unique = np.unique(ids[test_idx, 2])

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
