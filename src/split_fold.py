import pathlib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from utils import timer
from utils import load_pickle, dump_pickle

import config


def main():
    # ids, (site, floor, path)
    ids = load_pickle("../data/preprocessing/train_ids.pkl")

    path = ids[:, 2]
    cv = GroupKFold(n_splits=5)
    for n_fold, (train_idx, test_idx) in enumerate(cv.split(ids, groups=path)):
        print(f"Fold {n_fold:>02}")

        cv = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=config.SEED)
        for _train_idx, _valid_idx in cv.split(
            ids[train_idx], groups=ids[train_idx, 2]
        ):
            train_unique = np.unique(ids[_train_idx, 2])
            valid_unique = np.unique(ids[_valid_idx, 2])

        test_unique = np.unique(ids[test_idx, 2])

        print(
            f"\tTrain size: {len(_train_idx)}",
            f"Valid size: {len(_valid_idx)}",
            f"Test size: {len(test_idx)}",
        )
        print(
            f"\tTrain nunique: {len(train_unique)}",
            f"Valid nunique: {len(valid_unique)}",
            f"Test nunique: {len(test_unique)}",
        )

        np.save(f"../data/fold/fold{n_fold:>02}_train_idx.npy", _train_idx)
        np.save(f"../data/fold/fold{n_fold:>02}_valid_idx.npy", _valid_idx)
        np.save(f"../data/fold/fold{n_fold:>02}_test_idx.npy", test_idx)


if __name__ == "__main__":
    with timer("Split Data"):
        main()
