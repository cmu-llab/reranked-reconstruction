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

# === as if this returns search result ===

def get_hparam():
    match (d2p_run.config_class.dataset, p2d_run.config_class.architecture):
        # this is average from reranking_eval2
        case ('chinese_baxter', 'GRU'):
            return round(7.2), 1.7550000000000003
        case ('chinese_baxter', 'Transformer'):
            return round(6.50000001), 2.43
        case ('chinese_baxter', 'JambuTransformer'):
            return round(6.50000001), 2.415
        case ('chinese_wikihan2022', 'GRU'):
            return round(5.9), 1.395
        case ('chinese_wikihan2022', 'Transformer'):
            return round(6.3), 1.275
        case ('chinese_wikihan2022', 'JambuTransformer'):
            return round(6.6), 1.2600000000000002
        case ('chinese_wikihan2022_augmented', 'GRU'):
            return round(7.0), 1.6199999999999997
        case ('chinese_wikihan2022_augmented', 'Transformer'):
            return round(8.3), 1.7550000000000001
        case ('chinese_wikihan2022_augmented', 'JambuTransformer'):
            return round(6.6), 1.5749999999999997
        case ('Nromance_ipa', 'GRU'):
            return round(5.4), 0.41999999999999993
        case ('Nromance_ipa', 'Transformer'):
            return round(6.1), 0.5549999999999999
        case ('Nromance_ipa', 'JambuTransformer'):
            return round(6.0), 0.5850000000000001
        case ('Nromance_orto', 'GRU'):
            return round(6.1), 0.8699999999999999
        case ('Nromance_orto', 'Transformer'):
            return round(5.5000001), 0.99
        case ('Nromance_orto', 'JambuTransformer'):
            return round(5.2), 0.915
        case _:
            raise NotImplemented

# === evaluation! ===
    
best_beam_size, best_beam_reranker_weight_ratio = get_hparam()
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