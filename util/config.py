import argparse
import sys

from yacs.config import CfgNode as CN

_C = CN()
cfg = _C

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.IM_SIZE = 224
_C.TRAIN.DATASET = "./ImageNet/"
_C.TRAIN.SPLIT = "train"
_C.TRAIN.CHECKPOINT = None
_C.TRAIN.WORKERS = 4
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.PRINT_FEQ = 20

_C.TEST = CN()
_C.TEST.DATASET = "./ImageNet/"
_C.TEST.SPLIT = "val"
_C.TEST.BATCH_SIZE = 64
_C.TEST.IM_SIZE = 256
_C.TEST.PRINT_FEQ = 20

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True

_C.OPTIM = CN()
# Learning rate policy select from {'cos', 'steps'}
_C.OPTIM.LR_POLICY = "cos"
_C.OPTIM.LR_MULT = 0.1
_C.OPTIM.MAX_EPOCH = 100
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WARMUP_FACTOR = 0.1
_C.OPTIM.WARMUP_EPOCHS = 5
_C.OPTIM.WEIGHT_DECAY = 1e-4

_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    help_s = "LOCAL_RANK for torch.distributed.launch."
    parser.add_argument(
        "--local_rank", help=help_s, default=None, nargs=argparse.REMAINDER
    )
    help_s = "See utils/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)