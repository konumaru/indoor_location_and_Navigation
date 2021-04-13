class CommonConfig:
    SEED = 42

    NUM_FOLD = 5
    BATCH_SIZE = 512


class DebugConfig(CommonConfig):
    accelerator = None
    gpus = None
    NUM_EPOCH = 5

    def __init__(self):
        super(CommonConfig, self).__init__()


class Config(CommonConfig):
    accelerator = "dp"
    gpus = 1
    NUM_EPOCH = 200

    def __init__(self):
        super(CommonConfig, self).__init__()
