from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


class CommonConfig:
    SEED = 42

    NUM_FOLD = 5
    BATCH_SIZE = 512


class DebugConfig(CommonConfig):
    accelerator = None
    gpus = None
    NUM_EPOCH = 5
    DEV_RUN = 1

    callbacks = []
    logger = None

    def __init__(self):
        super(CommonConfig, self).__init__()


class Config(CommonConfig):
    accelerator = "dp"
    gpus = 1
    NUM_EPOCH = 200
    DEV_RUN = 0

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
    logger = TensorBoardLogger(save_dir="../tb_logs", name="Update-WifiModel")

    def __init__(self):
        super(CommonConfig, self).__init__()
