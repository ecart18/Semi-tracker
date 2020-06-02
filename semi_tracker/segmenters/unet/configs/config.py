
from __future__ import absolute_import

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "Cell segmentation using Unet"

__C.CUDA = True



# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18, 50, 101
__C.BACKBONE.TYPE = "uent"

__C.BACKBONE.KWARGS = CN(new_allowed=True)



# ------------------------------------------------------------------------ #
# LOSS options
# ------------------------------------------------------------------------ #
__C.LOSS = CN()

__C.LOSS.TYPE = 'CELoss'

__C.LOSS.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.SEED = 1

__C.TRAIN.RESUME = ""

__C.TRAIN.EPOCHS = 150

__C.TRAIN.START_SAVE_EPOCH = 0

__C.TRAIN.OPT = CN()

__C.TRAIN.OPT.TYPE = "Adam"

__C.TRAIN.OPT.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Test options
# ------------------------------------------------------------------------ #
__C.TEST = CN(new_allowed=True)
__C.TEST.CHECKPOINT_PATH = ""
__C.TEST.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)
__C.DATASET.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# TRACK options
# ------------------------------------------------------------------------ #
__C.TRACK = CN(new_allowed=True)
__C.TRACKER = CN(new_allowed=True)
__C.TRACKER.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# monitor options
# ------------------------------------------------------------------------ #
__C.MONITOR = CN(new_allowed=True)
__C.MONITOR.watch_dir_path = ""
__C.MONITOR.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# IO options
# ------------------------------------------------------------------------ #
__C.IO = CN(new_allowed=True)
__C.IO.LOG_DIR = './logs'



# num_instances' : help="each minibatch consist of batch_size // num_instances)
#  identities, and each identity has num_instances instances"