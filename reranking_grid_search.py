# run reranking grid search for a pair of d2p and p2d models

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

# === reranking tests ===

_ = d2p_model.eval(), p2d_model.eval()

def grid_search_on_val():
    independents = {
        'B': [2,4,6,8,10],
        'beam_reranker_weight_ratio': [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2],
    }
    experiments = []
    keys, values = zip(*independents.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = pd.DataFrame({
        'experiment_id': [],
        'beam_size': [],
        'beam_reranker_weight_ratio': [],
        'phoneme_edit_distance': [],
        'accuracy': [],
    })

    for experiment_id, experiment in tqdm(enumerate(experiments), total=len(experiments)):
        B = experiment['B']
        beam_reranker_weight_ratio = experiment['beam_reranker_weight_ratio']
        eval_res = batched_reranking_eval(d2p_model, p2d_model, d2p_dm, 
            reranker = BatchedCorrectRateReranker(p2d_model),
            rescorer = BatchedLinearRescorer(original_log_prob_weight = 1.0, reranker_weight = beam_reranker_weight_ratio), 
            beam_size = B, 
            split = 'val',
        )
        
        result = pd.DataFrame({
            'experiment_id': experiment_id,
            'beam_size': experiment['B'],
            'beam_reranker_weight_ratio': experiment['beam_reranker_weight_ratio'], 
            'phoneme_edit_distance': eval_res['d2p/test/phoneme_edit_distance'], # not the actual test set result, we just used val set as test set
            'accuracy': eval_res['d2p/test/accuracy'], # not the actual test set result, we just used val set as test set
        }, index=[experiment_id])

        results = pd.concat([results, result])
        
    print('grid search res:')
    print(results)

    # sorted_results = results.sort_values('phoneme_edit_distance', ascending=True)
    sorted_results = results.sort_values('accuracy', ascending=False)
    best_val = sorted_results.iloc[0]
    best_beam_size = int(best_val['beam_size'])
    best_beam_reranker_weight_ratio = best_val['beam_reranker_weight_ratio']
    
    return best_beam_size, best_beam_reranker_weight_ratio        
    
best_beam_size, best_beam_reranker_weight_ratio = grid_search_on_val()
run = wandb.init(
    mode = "online",
    entity = WANDB_ENTITY,
    project = WANDB_PROJECT,
    tags = ['reranking_eval2'],
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


