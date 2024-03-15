# region imports
from __future__ import annotations
from typing import TYPE_CHECKING
from lib import getfreegpu

from models.encoderDecoderTransformer import EncoderDecoderTransformer
from models.jambuGRU import JambuGRU
from models.jambuTransformer import JambuTransformer
from matplotlib import pyplot as plt
import wandb
api = wandb.Api()
from lib.dataloader_manager import DataloaderManager
from models.encoderDecoderRNN import Seq2SeqRNN
import torch
import pytorch_lightning as pl
from torch import Tensor
from specialtokens import *
from torch.nn import functional as F
from tqdm.notebook import tqdm
from einops import rearrange, repeat
import pandas as pd
from lib.tensor_utils import num_sequences_equal, sequences_equal
from lib.rescoring import BatchedCorrectRateReranker, BatchedGreedyBasedRerankerBase, BatchedLinearRescorer, BatchedRandomReranker, BeamPositionReranker, CorrectRateReranker, LinearRescorer, CharEditDistanceReranker, PhonemeEditDistanceReranker, HybridGreedyBasedReranker, RandomScoreReranker
import seaborn as sns
from prelude import *
import warnings
warnings.filterwarnings(action="ignore", message=".*num_workers.*")
warnings.filterwarnings(action="ignore", message=".*negatively affect performance.*")
warnings.filterwarnings(action="ignore", message=".*MPS available.*")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import os
from dotenv import load_dotenv
load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
import pickle
# endregion

TEST_VAL_BATCH_SIZE = 64 # unified batch size for analysis inference

def is_sorted_dec(l):
    return all(l[i] >= l[i+1] for i in range(len(l) - 1))

# trick to get config from dictionary with . syntax
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SubmodelRun:
    def __init__(self, run_id, version, verbose: bool = False):
        artifact = api.artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/model-{run_id}:{version}')
        artifact_dir = artifact.download()
        if verbose: print(f"artifact sits in: {artifact_dir}")
        
        run = api.run(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}')
        config_dict = run.config
        
        default_config_dict = {
            'proportion_labelled': 1.0,
            'warmup_epochs': 0, # doesn't matter for eval
            'max_epochs': 100, # doesn't matter for eval
            'weight_decay': 0.0, # doesn't matter for eval
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
        }
        
        for k, v in default_config_dict.items():
            if not k in config_dict:
                config_dict[k] = v

        config_class = AttributeDict(config_dict)
        
        self.artifact, self.artifact_dir, self.run, self.config_dict, self.config_class = artifact, artifact_dir, run, config_dict, config_class
       
    def get_dm(self):
        dm = DataloaderManager(
            data_dir = f"data/{self.config_class.dataset}", 
            
            batch_size = self.config_class.batch_size,
            test_val_batch_size = TEST_VAL_BATCH_SIZE,
            shuffle_train = True,
            
            lang_separators = self.config_class.d2p_use_lang_separaters,
            skip_daughter_tone = self.config_class.skip_daughter_tone,
            skip_protoform_tone = self.config_class.skip_protoform_tone,
            include_lang_tkns_in_ipa_vocab = True,
            
            daughter_subset = None if self.config_class.daughter_subset == "None" else self.config_class.daughter_subset,
            min_daughters = self.config_class.min_daughters,
            verbose = False,
            
            transformer_d2p_d_cat_style = (self.config_class.architecture == "Transformer"),
            proportion_labelled = self.config_class.proportion_labelled,
        )
        return dm

    def get_model(self, dm: DataloaderManager):
        match (self.config_class.submodel, self.config_class.architecture):
            case ('d2p', 'GRU'):
                model = Seq2SeqRNN.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),
                    
                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab, 
                    
                    num_encoder_layers = self.config_class.d2p_num_encoder_layers,
                    dropout_p = self.config_class.d2p_dropout_p,
                    use_vae_latent = self.config_class.d2p_use_vae_latent,
                    inference_decode_max_length = self.config_class.d2p_inference_decode_max_length,
                    use_bidirectional_encoder = self.config_class.d2p_use_bidirectional_encoder,
                    decode_mode = self.config_class.d2p_decode_mode,
                    beam_search_alpha = self.config_class.d2p_beam_search_alpha,
                    beam_size = self.config_class.d2p_beam_size,
                    lang_embedding_when_decoder = self.config_class.d2p_lang_embedding_when_decoder,

                    feedforward_dim = self.config_class.d2p_feedforward_dim, 
                    embedding_dim = self.config_class.d2p_embedding_dim,
                    model_size = self.config_class.d2p_model_size,

                    init_embedding = None,
                    training_mode = 'encodeWithLangEmbedding',
                    logger_prefix = 'd2p',
                    
                    encoder_takes_prev_decoder_out=False,
                    prompt_mlp_with_one_hot_lang=False,
                    gated_mlp_by_target_lang=False,
                    all_lang_summary_only = True,

                    use_xavier_init = False,
                    lr = self.config_class.lr,
                    warmup_epochs = self.config_class.warmup_epochs,
                    max_epochs = self.config_class.max_epochs,
                    
                    beta1 = self.config_class.beta1,
                    beta2 = self.config_class.beta2,
                    eps = self.config_class.eps,
                )
            case ('p2d', 'GRU'):
                model = Seq2SeqRNN.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),
                    
                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab, 
                    
                    num_encoder_layers = self.config_class.p2d_num_encoder_layers,
                    dropout_p = self.config_class.p2d_dropout_p,
                    use_vae_latent = self.config_class.p2d_use_vae_latent,
                    inference_decode_max_length = self.config_class.p2d_inference_decode_max_length,
                    use_bidirectional_encoder = self.config_class.p2d_use_bidirectional_encoder,
                    decode_mode = self.config_class.p2d_decode_mode,
                    beam_search_alpha = None if self.config_class.p2d_beam_search_alpha == "None" else self.config_class.p2d_beam_search_alpha, # workaround for wandb bug https://github.com/wandb/wandb/issues/4016
                    beam_size = None if self.config_class.p2d_beam_size == "None" else self.config_class.p2d_beam_size, # workaround for wandb bug https://github.com/wandb/wandb/issues/4016
                    lang_embedding_when_decoder = self.config_class.p2d_lang_embedding_when_decoder,
                    prompt_mlp_with_one_hot_lang = self.config_class.p2d_prompt_mlp_with_one_hot_lang,
                    gated_mlp_by_target_lang = self.config_class.p2d_gated_mlp_by_target_lang,

                    feedforward_dim = self.config_class.p2d_feedforward_dim, 
                    embedding_dim = self.config_class.p2d_embedding_dim,
                    model_size = self.config_class.p2d_model_size,

                    init_embedding = None,
                    training_mode = 'encodeTokenSeqAppendTargetLangToken',
                    logger_prefix = 'p2d',
                    
                    encoder_takes_prev_decoder_out=False,
                    all_lang_summary_only  = self.config_class.p2d_all_lang_summary_only,

                    use_xavier_init = False,
                    lr = self.config_class.lr,
                    warmup_epochs = self.config_class.warmup_epochs,
                    max_epochs = self.config_class.max_epochs,

                    beta1 = self.config_class.beta1,
                    beta2 = self.config_class.beta2,
                    eps = self.config_class.eps,
                )
            case ('d2p', 'Transformer'):
                model = EncoderDecoderTransformer.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),
                    
                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab,
                    num_encoder_layers = self.config_class.d2p_num_encoder_layers,
                    num_decoder_layers = self.config_class.d2p_num_decoder_layers,
                    embedding_dim = self.config_class.d2p_embedding_dim,
                    nhead = self.config_class.d2p_nhead,
                    feedforward_dim = self.config_class.d2p_feedforward_dim,
                    dropout_p = self.config_class.d2p_dropout_p,
                    max_len = self.config_class.d2p_max_len,
                    logger_prefix = 'd2p',
                    task = 'd2p',
                    inference_decode_max_length = self.config_class.d2p_inference_decode_max_length,
                    all_lang_summary_only = True,
                    
                    use_xavier_init = False,
                    lr = self.config_class.lr,
                    warmup_epochs = self.config_class.warmup_epochs,
                    max_epochs = self.config_class.max_epochs,
                    weight_decay = self.config_class.weight_decay,

                    beta1 = self.config_class.beta1,
                    beta2 = self.config_class.beta2,
                    eps = self.config_class.eps,
                )
            case ('p2d', 'Transformer'):
                model = EncoderDecoderTransformer.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),
                    
                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab,
                    num_encoder_layers = self.config_class.p2d_num_encoder_layers,
                    num_decoder_layers = self.config_class.p2d_num_decoder_layers,
                    embedding_dim = self.config_class.p2d_embedding_dim,
                    nhead = self.config_class.p2d_nhead,
                    feedforward_dim = self.config_class.p2d_feedforward_dim,
                    dropout_p = self.config_class.p2d_dropout_p,
                    max_len = self.config_class.p2d_max_len,
                    logger_prefix = 'p2d',
                    task = 'p2d',
                    inference_decode_max_length = self.config_class.p2d_inference_decode_max_length,
                    all_lang_summary_only = self.config_class.p2d_all_lang_summary_only,
                    
                    use_xavier_init = False,
                    lr = self.config_class.lr,
                    warmup_epochs = self.config_class.warmup_epochs,
                    max_epochs = self.config_class.max_epochs,
                    weight_decay = self.config_class.weight_decay,

                    beta1 = self.config_class.beta1,
                    beta2 = self.config_class.beta2,
                    eps = self.config_class.eps,
                )
            case ('p2d', 'JambuGRU'):
                model = JambuGRU.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),
                    
                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab,
                    emb_size = self.config_class.emb_size,
                    hidden_size = self.config_class.hidden_size,
                    num_layers = self.config_class.num_layers,
                    dropout = self.config_class.dropout,
                    bidirectional = self.config_class.bidirectional,
                    lr = self.config_class.lr,
                    logger_prefix = 'p2d',
                    inference_decode_max_length = self.config_class.p2d_inference_decode_max_length,
                    all_lang_summary_only = self.config_class.p2d_all_lang_summary_only,
                    warmup_epochs = self.config_class.warmup_epochs,
                    max_epochs = self.config_class.max_epochs,
                )
            case ('p2d', 'JambuTransformer'):
                model = JambuTransformer.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),
                    
                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab,
                    n_layers = self.config_class.n_layers,
                    d_model = self.config_class.d_model,
                    d_ff = self.config_class.d_ff,
                    n_heads = self.config_class.n_heads,
                    dropout = self.config_class.dropout,
                    lr = self.config_class.lr,
                    logger_prefix = 'p2d',
                    inference_decode_max_length = self.config_class.p2d_inference_decode_max_length,
                    all_lang_summary_only = self.config_class.p2d_all_lang_summary_only,
                    warmup_epochs = self.config_class.warmup_epochs,
                    max_epochs = self.config_class.max_epochs,
                )
            case _:
                raise ValueError(f"Unknown model type")
        model.eval()
        return model

class SubmodelRunFromFile(SubmodelRun):
    def __init__(self, checkpoint_path: str, config_path: str, verbose: bool = False):
        
        artifact_dir = checkpoint_path
        config_dict = pickle.load(open(config_path, 'rb'))
        
        default_config_dict = {
            'proportion_labelled': 1.0,
            'warmup_epochs': 0, # doesn't matter for eval
            'max_epochs': 100, # doesn't matter for eval
            'weight_decay': 0.0, # doesn't matter for eval
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
        }
        
        for k, v in default_config_dict.items():
            if not k in config_dict:
                config_dict[k] = v

        config_class = AttributeDict(config_dict)
        
        self.artifact, self.artifact_dir, self.run, self.config_dict, self.config_class = None, artifact_dir, None, config_dict, config_class




# note: evaluation will treat whatever split as the test set
def eval_on_set(submodel, dm, split: str):
    evaluator = pl.Trainer(
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices= getfreegpu.assign_free_gpus(threshold_vram_usage=2000, max_gpus=1, wait=True, sleep_time=10) if torch.cuda.is_available() else 'auto',
        max_epochs=1,
    )
    match split:
        case 'val':
            res = evaluator.test(submodel, dataloaders=dm.val_dataloader(), verbose=False)
        case 'train':
            res = evaluator.test(submodel, dataloaders=dm.train_dataloader(), verbose=False)
        case 'test':
            res = evaluator.test(submodel, dataloaders=dm.test_dataloader(), verbose=False)
        case _:
            raise ValueError(f"Unknown split: {split}")
    return res[0]

def get_sample_batch(dm, split: str):
    if split == 'test':
        for batch in dm.test_dataloader():       
            return batch
    if split == 'train':
        for batch in dm.train_dataloader():       
            return batch
    if split == 'val':
        for batch in dm.val_dataloader():       
            return batch
    raise ValueError(f"Unknown split: {split}")

def get_accuracy(model, dm: DataloaderManager, split: str):
    evaluator = pl.Trainer(accelerator='cpu', max_epochs=1, enable_progress_bar=False)
    loader = get_loader_for_split(dm, split)
    accuracy = evaluator.validate(model, dataloaders=loader, verbose=False)[0][f'{model.logger_prefix}/val/accuracy']
    return accuracy

def get_loader_for_split(dm, split):
    match split:
        case 'train':
            loader = dm.train_dataloader()
        case 'val':
            loader = dm.val_dataloader()
        case 'test':
            loader = dm.test_dataloader()
        case _:
            raise ValueError(f'Invalid split: {split}')
    return loader

def batched_reranking_eval(d2p_model: Seq2SeqRNN, p2d_model: Seq2SeqRNN, d2p_dm, reranker, rescorer, beam_size, split: str):
    _ = d2p_model.eval, p2d_model.eval()
    
    d2p_model.reranker = reranker
    d2p_model.rescorer = rescorer

    tmp = d2p_model.decode_mode, d2p_model.beam_size
    d2p_model.decode_mode = 'batched_reranked_beam_search'
    d2p_model.beam_size = beam_size
    
    res = eval_on_set(d2p_model, d2p_dm, split)
    d2p_model.decode_mode, d2p_model.beam_size = tmp
    
    return res

def beam_search_eval(model: Seq2SeqRNN, dm, beam_size, split: str):
    assert (isinstance(model, Seq2SeqRNN))
    tmp = model.decode_mode, model.beam_size
    model.decode_mode = 'beam_search'
    model.beam_size = beam_size
    
    res = eval_on_set(model, dm, split)
    model.decode_mode, model.beam_size = tmp
    
    return res
