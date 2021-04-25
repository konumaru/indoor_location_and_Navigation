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

from utils.common import timer
from utils.common import load_pickle

from dataset import IndoorDataModule
from models import InddorModel, MeanPositionLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
    parser.add_argument("-v", "--valid", help="validation mode", action="store_true")
    args = parser.parse_args()

    if args.debug:
        from config import DebugConfig as Config
    elif args.valid:
        from config import ValidConfig as Config
    else:
        from config import Config as Config

    pl.seed_everything(Config.SEED)
    for n_fold in range(Config.NUM_FOLD):
        # Load index and select fold daata.
        train_idx = np.load(f"../data/fold/fold{n_fold:>02}_train_idx.npy")
        valid_idx = np.load(f"../data/fold/fold{n_fold:>02}_valid_idx.npy")
        test_idx = np.load(f"../data/fold/fold{n_fold:>02}_test_idx.npy")

        # Define and setup datamodule.
        datamodule = IndoorDataModule(Config.BATCH_SIZE, train_idx, valid_idx, test_idx)
        datamodule.setup()

        checkpoint_callback = ModelCheckpoint(monitor="valid_loss", mode="min")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            min_delta=0.01,
            patience=10,
            verbose=False,
            mode="min",
        )

        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
        logger = TensorBoardLogger(save_dir="../tb_logs", name="Test")

        model = InddorModel(lr=1e-3)
        trainer = Trainer(
            accelerator=Config.accelerator,
            gpus=Config.gpus,
            max_epochs=Config.NUM_EPOCH,
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=Config.DEV_RUN,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

        print(checkpoint_callback.best_model_path)

        break


if __name__ == "__main__":
    with timer("Train"):
        main()
