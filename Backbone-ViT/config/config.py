from yacs.config import CfgNode as CN
cfg = CN()

cfg.SEED = 0

# dataset path
cfg.DATA_PATH = 'DATASET_PATH'    

cfg.PRETRAIN_PATH = 'model\\jx_vit_base_p16_224-80ecf9dd.pth'

cfg.START_EPOCH = 1
cfg.MAX_EPOCH = 80

cfg.H = 256
cfg.W = 128
cfg.BATCH_SIZE = 32  # num of images for each modality in a mini batch
cfg.NUM_POS = 4

cfg.MARGIN = 0.1    # margin for triplet

# model
cfg.STRIDE_SIZE =  [12,12]
cfg.DROP_OUT = 0.07
cfg.ATT_DROP_RATE = 0.07
cfg.DROP_PATH = 0.08

# optimizer
cfg.OPTIMIZER_NAME = 'SGD'  # AdamW or SGD
cfg.MOMENTUM = 0.9    # for SGD

cfg.BASE_LR = 7.5e-5
cfg.WEIGHT_DECAY = 7e-3
cfg.WEIGHT_DECAY_BIAS = 7e-3
cfg.BIAS_LR_FACTOR = 1

cfg.LR_PRETRAIN = 0.1
cfg.LR_MIN = 0.1
cfg.LR_INIT = 0.1
cfg.WARMUP_EPOCHS = 5








