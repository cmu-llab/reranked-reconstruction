# run reranking test suite for a pair of d2p and p2d models, with direct hparam input

import torch
from einops import rearrange, repeat

import pytorch_lightning as pl
from lib.analysis_utils import *
from specialtokens import *
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import torchshow as ts
import os
import argparse
import json
from lib.reranking_utils import get_reranking_hparam
import os
from dotenv import load_dotenv
load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

parser = argparse.ArgumentParser()
parser.add_argument("d2p_id") # run id 
parser.add_argument("p2d_id") # run id
args = parser.parse_args()
config = vars(args)

print("running: ", config)

D2P_RUN = config["d2p_id"]
P2D_RUN = config["p2d_id"]
   
# === load d2p model ===

d2p_run = SubmodelRun(D2P_RUN, 'best')
assert d2p_run.config_class.submodel == 'd2p'
d2p_dm = d2p_run.get_dm()
d2p_model = d2p_run.get_model(d2p_dm)

# === load p2d model ===

p2d_run = SubmodelRun(P2D_RUN, 'best')
assert p2d_run.config_class.submodel == 'p2d'
p2d_dm = p2d_run.get_dm()
p2d_model = p2d_run.get_model(p2d_dm)
   
# === Making sure the two models use same dataset ===
    
assert d2p_run.config_class.dataset == p2d_run.config_class.dataset

# === evaluation! ===
    
best_beam_size, best_beam_reranker_weight_ratio = get_reranking_hparam(d2p_run, p2d_run)
run = wandb.init(
    mode = "online",
    entity = WANDB_ENTITY,
    project = WANDB_PROJECT,
    tags = ['reranking_eval2_fixed_beam_fixed_ratio'],
    config = {
        'd2p_run': D2P_RUN,
        'd2p_architecture': d2p_run.config_class.architecture,
        'p2d_run': P2D_RUN,
        'p2d_architecture': p2d_run.config_class.architecture,
        'dataset': d2p_run.config_class.dataset,
        'best_beam_size': best_beam_size,
        'best_beam_reranker_weight_ratio': best_beam_reranker_weight_ratio,
    },
)
print('this run:', wandb.config)
test_res = batched_reranking_eval(d2p_model, p2d_model, d2p_dm, 
    reranker = BatchedCorrectRateReranker(p2d_model),
    rescorer = BatchedLinearRescorer(original_log_prob_weight = 1.0, reranker_weight = best_beam_reranker_weight_ratio), 
    beam_size = best_beam_size, 
    split = 'test',
)
print('result:', test_res)
wandb.log(test_res)