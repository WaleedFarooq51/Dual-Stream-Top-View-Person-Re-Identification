from yacs.config import CfgNode as CN

cfg = CN()

cfg.DATA_PATH= 'DATA_DIRECTORY_PATH'

cfg.MAX_EPOCH= 65

cfg.BASE_LR = 5e-5

cfg.LR_MIN = 0.1
cfg.LR_INIT = 0.1
cfg.WARMUP_EPOCHS = 5








