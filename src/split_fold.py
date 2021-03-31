import pathlib
import numpy as np
from sklearn.model_selection import GroupKFold

from utils import timer
from utils import load_pickle, dump_pickle


def main():
    # ids, (site, floor, path)
    ids = load_pickle("../data/preprocessing/train_ids.pkl")

    path = ids[:, 2]
    cv = GroupKFold(n_splits=5)
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(ids, groups=path)):
        print(f"Fold {n_fold:>02}")

        train_unique = np.unique(ids[train_idx, 2])
        valid_unique = np.unique(ids[valid_idx, 2])

        print(f"\tTrain size: {len(train_idx)} Valid size: {len(valid_idx)}")
        print(
            f"\tTrain nunique: {len(train_unique)} Valid nunique: {len(valid_unique)}"
        )

        np.save(f"../data/fold/fold{n_fold:>02}_train_idx.npy", train_idx)
        np.save(f"../data/fold/fold{n_fold:>02}_valid_idx.npy", valid_idx)


if __name__ == "__main__":
    with timer("Split Data"):
        main()
