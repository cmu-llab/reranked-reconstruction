# rerun test epoch on a d2p checkpoint with a different beam size

from lib.analysis_utils import *
from specialtokens import *
from tqdm import tqdm
import pandas as pd
import itertools
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

parser = argparse.ArgumentParser()
parser.add_argument("d2p_id") # run id 
args = parser.parse_args()
config = vars(args)
new_beam_size = 10

print("running: ", config)

D2P_RUN = config["d2p_id"]
    
# === load d2p model ===

d2p_run = SubmodelRun(D2P_RUN, 'best')
assert d2p_run.config_class.submodel == 'd2p'
d2p_dm = d2p_run.get_dm()
d2p_model = d2p_run.get_model(d2p_dm)

# === rerun with different beam size ===

_ = d2p_model.eval()

test_res = beam_search_eval(d2p_model, d2p_dm, new_beam_size, 'test')
print(test_res)

# === log to wandb ===
    
run = wandb.init(
    mode = "online",
    entity = WANDB_ENTITY,
    project = WANDB_PROJECT,
    tags = ['beam_size_adjustment'],
    config = {
        'd2p_run': D2P_RUN,
        'd2p_architecture': d2p_run.config_class.architecture,
        'dataset': d2p_run.config_class.dataset,
        'new_beam_size': new_beam_size,
    },
)
print('this run:', wandb.config)
print('result:', test_res)
wandb.log(test_res)