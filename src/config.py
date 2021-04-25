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


class ValidConfig(CommonConfig):
    accelerator = "dp"
    gpus = 1
    NUM_EPOCH = 20
    DEV_RUN = 0

    def __init__(self):
        super(CommonConfig, self).__init__()


class Config(CommonConfig):
    accelerator = "dp"
    gpus = 1
    NUM_EPOCH = 200
    DEV_RUN = 0

    def __init__(self):
        super(CommonConfig, self).__init__()
