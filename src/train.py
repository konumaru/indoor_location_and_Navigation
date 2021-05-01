import re
import shutil
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from typing import List, AnyStr

from utils.common import timer
from utils.common import load_pickle

from dataset import IndoorDataModule
from models import InddorModel, MeanPositionLoss


def get_config(mode: str):
    if mode == "debug":
        from config import DebugConfig as config
    elif mode == "valid":
        from config import ValidConfig as config
    elif mode == "train":
        from config import Config as config
    return config


def dump_cv_metric(model_name: str, version: int, metric: float):
    with open("../checkpoints/scores.txt", "a") as f:
        txt = f"\n{model_name: >24} {str(version): >4} {metric:.4f}"
        f.write(txt)


def dump_best_checkpoints(best_checkpoints: List, model_name: AnyStr):
    with open(f"../checkpoints/{model_name}.txt", "w") as f:
        txt = "\n".join(best_checkpoints)
        f.write(txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, choices=["debug", "valid", "train"], default="debug"
    )
    parser.add_argument("-n", "--model_name", type=str, default="Debug", required=True)
    parser.add_argument("-v", "--version", type=int, default=0, required=True)
    args = parser.parse_args()

    config = get_config(args.mode)
    pl.seed_everything(config.SEED)

    metrics = []
    best_checkpoints = []
    for n_fold in range(config.NUM_FOLD):
        # Load index and select fold daata.
        train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
        valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")
        test_idx = np.load(f"../data/fold/fold{n_fold:>02}_test_idx.npy")

        # Define and setup datamodule.
        datamodule = IndoorDataModule(config.BATCH_SIZE, train_idx, valid_idx, test_idx)
        datamodule.setup()

        checkpoint_callback = ModelCheckpoint(
            filename="{epoch:02d}-{trian_loss:.4f}-{valid_loss:.4f}",
            monitor="valid_loss",
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            patience=10,
            verbose=False,
            mode="min",
        )

        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
        logger = TensorBoardLogger(
            save_dir="../tb_logs",
            name=args.model_name,
            version=f"0.{args.version}.{n_fold}",
        )

        model = InddorModel(lr=1e-3)
        trainer = Trainer(
            accelerator=config.accelerator,
            gpus=config.gpus,
            max_epochs=config.NUM_EPOCH,
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=config.DEV_RUN,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

        best_checkpoints.append(checkpoint_callback.best_model_path)

        if bool(config.DEV_RUN):
            break
        else:
            metric = float(
                re.findall(
                    r"valid_loss=(\d+\.\d+)", checkpoint_callback.best_model_path
                )[0]
            )
            metrics.append(metric)

    dump_cv_metric(args.model_name, args.version, np.mean(metrics))
    dump_best_checkpoints(best_checkpoints, args.model_name)


if __name__ == "__main__":
    with timer("Train"):
        main()
