import torch
from einops import rearrange, repeat
import pytorch_lightning as pl
from specialtokens import *
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
import itertools
import os
import json
from torch import Tensor

torch.set_printoptions(precision=4)
torch.set_printoptions(sci_mode=False)

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


rescored_endnodes_t = tuple[Tensor, Tensor, Tensor, Tensor, Tensor] 
batch_t = tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
ALL_TARGET_LANGS_LABEL: str = 'all_target_langs'
