#! pretrains a model to perform either d2p (reconstruction) or p2d (reflex prediction)

# region === imports ===
import wandb
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pickle, os
import torchshow as ts
from lib.dataloader_manager import DataloaderManager
from models.encoderDecoderTransformer import EncoderDecoderTransformer
from models.encoderDecoderRNN import Seq2SeqRNN
from models.jambuGRU import JambuGRU
from models.jambuTransformer import JambuTransformer
import logging
import random
logging.getLogger('lingpy').setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_printoptions(precision=4)
torch.set_printoptions(sci_mode=False)
import warnings
warnings.filterwarnings(action="ignore", message=".*num_workers.*")
warnings.filterwarnings(action="ignore", message=".*negatively affect performance.*")
warnings.filterwarnings(action="ignore", message=".*MPS available.*")
import lib.getfreegpu
from pytorch_lightning.callbacks.callback import Callback
import argparse
from dotenv import load_dotenv
load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
# endregion


# region === run info ===
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False) # if specified, overrides all config, else use the config below or the config in the sweep
parser.add_argument('--dev', action='store_true')
parser.add_argument('--nowandb', action='store_true')
parser.add_argument('--wandbwatch', action='store_true')
parser.add_argument('--sweeping', action='store_true')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--d2p', action='store_true')
parser.add_argument('--p2d', action='store_true')
parser.add_argument('--GRU', action='store_true')
parser.add_argument('--Transformer', action='store_true')
parser.add_argument('--JambuGRU', action='store_true')
parser.add_argument('--JambuTransformer', action='store_true')
args = parser.parse_args()
print(args)
seed = random.randint(0, 4294967295) # range accepted by numpy
pl.seed_everything(seed)

nowandb = args.nowandb
cpu = args.cpu
if args.config is not None:
    loaded_config = pickle.load(open(args.config, 'rb'))
    dev = False
    wandbwatch = False
    sweeping = False
    submodel = loaded_config['submodel']
    architecture = loaded_config['architecture']
else:
    dev = args.dev
    wandbwatch = args.wandbwatch
    sweeping = args.sweeping
    match ('--p2d' in sys.argv, '--d2p' in sys.argv):
        case (True, False):
            submodel = 'p2d'
        case (False, True):
            submodel = 'd2p'
        case _:
            raise ValueError("You must specify exactly one of --p2d or --d2p")
    match ('--GRU' in sys.argv, '--Transformer' in sys.argv, '--JambuGRU' in sys.argv, '--JambuTransformer' in sys.argv):
        case (True, False, False, False):
            architecture = 'GRU'
        case (False, True, False, False):
            architecture = 'Transformer'
        case (False, False, True, False):
            architecture = 'JambuGRU'
        case (False, False, False, True):
            architecture = 'JambuTransformer'
        case _:
            raise ValueError("You must specify exactly one of --GRU, --Transformer, --JambuGRU, or --JambuTransformer")

wandb_name = ""
wandb_tags = [architecture, submodel]
if args.config is not None:
    wandb_tags.append('from-config')
wandb_notes = ""
set_devices = 'auto' if cpu else lib.getfreegpu.assign_free_gpus(threshold_vram_usage=1000, max_gpus=1, wait=True, sleep_time=10)
# endregion


# region === default hyperparams (if not using config) ===
config = {
    'submodel': submodel,
    'seed': seed,
    'architecture': architecture,
    'dataset': 'chinese_wikihan2022',
    'batch_size': 64,
    'use_xavier_init': True,
    'lr': 0.00013,
    'warmup_epochs': 50,
    'd2p_use_lang_separaters': True, # this doesn't matter for p2d
    'p2d_all_lang_summary_only': True, # this doesn't matter for d2p
    'proportion_labelled': 1.0, # fixed at 1.0 for supervised reconstruction
    'max_epochs': 200,
    'check_val_every_n_epoch': 5,
    'beta1': 0.9, # effective on encoderDecoderRNN and encoderDecoderTransformer
    'beta2': 0.999, # effective on encoderDecoderRNN and encoderDecoderTransformer
    'eps': 1e-8, # effective on encoderDecoderRNN and encoderDecoderTransformer
}
if (not sweeping) and (config['proportion_labelled'] < 1.0):
    assert submodel == 'd2p'
    wandb_tags.append(str(config['proportion_labelled']) + 'labelled')
    wandb_tags.append('semisupervised-exps')
d2p_dataset_config = {
    'daughter_subset': None,
    'min_daughters': 3,
    'skip_daughter_tone': False,
    'skip_protoform_tone': False,
}
p2d_dataset_config = {
    'daughter_subset': None,
    'min_daughters': 3,
    'skip_daughter_tone': False,
    'skip_protoform_tone': False,   
}
GRU_d2p_config = {    
    'd2p_num_encoder_layers': 2,
    'd2p_dropout_p': 0.30,
    'd2p_use_vae_latent': False,
    'd2p_inference_decode_max_length': 12,
    'd2p_use_bidirectional_encoder': False,
    'd2p_decode_mode': 'greedy_search',
    'd2p_beam_search_alpha': 1.0,
    'd2p_beam_size': 3 if not dev else 3,
    'd2p_lang_embedding_when_decoder': True,
    'd2p_feedforward_dim': 512 if not dev else 15,
    'd2p_embedding_dim': 300 if not dev else 16,
    'd2p_model_size': 128 if not dev else 17,
}
GRU_p2d_config = {
    'p2d_num_encoder_layers': 2,
    'p2d_dropout_p': 0.30,
    'p2d_use_vae_latent': False,
    'p2d_inference_decode_max_length': 12,
    'p2d_use_bidirectional_encoder': True,
    'p2d_decode_mode': 'greedy_search',
    'p2d_beam_search_alpha': 1.0,
    'p2d_beam_size': 5 if not dev else 3,
    'p2d_lang_embedding_when_decoder': False,
    'p2d_prompt_mlp_with_one_hot_lang': True, 
    'p2d_gated_mlp_by_target_lang': False,
    'p2d_feedforward_dim': 512 if not dev else 16,
    'p2d_embedding_dim': 300 if not dev else 16,
    'p2d_model_size': 128 if not dev else 16,   
}
Transformer_d2p_config = {
    'd2p_num_encoder_layers': 1 if dev else 3,
    'd2p_num_decoder_layers': 1 if dev else 3,
    'd2p_embedding_dim': 32 if dev else 128,
    'd2p_nhead': 8,
    'd2p_feedforward_dim': 32 if dev else 128,
    'd2p_dropout_p': 0.202,
    'd2p_max_len': 30,
    'd2p_inference_decode_max_length': 30,
    'weight_decay': 0.0,
}
Transformer_p2d_config = {
    'p2d_num_encoder_layers': 1 if dev else 3,
    'p2d_num_decoder_layers': 1 if dev else 3,
    'p2d_embedding_dim': 32 if dev else 128,
    'p2d_nhead': 8,
    'p2d_feedforward_dim': 32 if dev else 128,
    'p2d_dropout_p': 0.2,
    'p2d_max_len': 32 if dev else 128,
    'p2d_inference_decode_max_length': 12,
    'weight_decay': 0.0,
}
Jambu_GRU_p2d_config = {
    'emb_size': 256,
    'hidden_size': 512,
    'num_layers': 1,
    'dropout': 0.1,
    'bidirectional': True,
    'p2d_inference_decode_max_length': 12,
}
JambuTransformer_p2d_config = {
    'n_layers': 2, 
    'd_model': 128, 
    'd_ff': 1024, 
    'n_heads': 8, 
    'dropout': 0.2,
    'p2d_inference_decode_max_length': 12,
}

if args.config is not None:
    config = {**loaded_config, **{'seed': seed}}
else:
    match (submodel, architecture):
        case ('d2p', 'GRU'):
            config = {**config, **d2p_dataset_config, **GRU_d2p_config}
        case ('d2p', 'Transformer'):
            config = {**config, **d2p_dataset_config, **Transformer_d2p_config}
        case ('p2d', 'GRU'):
            config = {**config, **p2d_dataset_config, **GRU_p2d_config}
        case ('p2d', 'Transformer'):
            config = {**config, **p2d_dataset_config, **Transformer_p2d_config}
        case ('p2d', 'JambuGRU'):
            config = {**config, **p2d_dataset_config, **Jambu_GRU_p2d_config}
        case ('p2d', 'JambuTransformer'):
            config = {**config, **p2d_dataset_config, **JambuTransformer_p2d_config}
        case _:
            raise ValueError(f"Unknown submodel {submodel} and architecture {architecture}")
# endregion


# region === wandb ===
if sweeping:
    # in that case we use sweepe config instead. this ensures nothing defaults.
    config = {
        'seed': seed,
        'submodel': submodel,
        'architecture': architecture,
    } 
run = wandb.init(
    mode = "disabled" if nowandb else "online",
    entity = WANDB_ENTITY,
    project = WANDB_PROJECT, 
    config = config,
    tags=wandb_tags,
    notes=wandb_notes if wandb_notes != "" else None,
    allow_val_change=True,
    name=wandb_name if wandb_name != "" else None,
)
wandb_logger = WandbLogger(
    log_model = True if not sweeping else False,
    experiment = run,
)
print(f"\nFinal wandb.config: {wandb.config}\n")

wandb.define_metric(f'{submodel}/val/accuracy', summary='max')
wandb.define_metric(f'{submodel}/val/target_in_beam', summary='max')
wandb.define_metric(f'{submodel}/val/bcubed_f_score', summary='max')
wandb.define_metric(f'{submodel}/val/phoneme_edit_distance', summary='min')
wandb.define_metric(f'{submodel}/val/loss', summary='min')

wandb.define_metric(f'{submodel}/test/accuracy')
wandb.define_metric(f'{submodel}/test/target_in_beam')
wandb.define_metric(f'{submodel}/test/bcubed_f_score')
wandb.define_metric(f'{submodel}/test/phoneme_edit_distance')
wandb.define_metric(f'{submodel}/test/loss')
# endregion


# region === loading model and data ===
dm = DataloaderManager(
    data_dir = f"data/{wandb.config.dataset}", 
    
    batch_size = wandb.config.batch_size,
    test_val_batch_size = 128,
    shuffle_train = True,
    
    lang_separators = wandb.config.d2p_use_lang_separaters,
    skip_daughter_tone = wandb.config.skip_daughter_tone,
    skip_protoform_tone = wandb.config.skip_protoform_tone,
    include_lang_tkns_in_ipa_vocab = True,
    transformer_d2p_d_cat_style = architecture == 'Transformer',
    
    daughter_subset = None if wandb.config.daughter_subset == "None" else wandb.config.daughter_subset,
    min_daughters = wandb.config.min_daughters,
    verbose = False,
    
    proportion_labelled = wandb.config.proportion_labelled, 
)

match (submodel, architecture):
    case ('d2p', 'GRU'):
        model = Seq2SeqRNN(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab, 
            
            num_encoder_layers = wandb.config.d2p_num_encoder_layers,
            dropout_p = wandb.config.d2p_dropout_p,
            use_vae_latent = wandb.config.d2p_use_vae_latent,
            inference_decode_max_length = wandb.config.d2p_inference_decode_max_length,
            use_bidirectional_encoder = wandb.config.d2p_use_bidirectional_encoder,
            decode_mode = wandb.config.d2p_decode_mode,
            beam_search_alpha = wandb.config.d2p_beam_search_alpha,
            beam_size = wandb.config.d2p_beam_size,
            lang_embedding_when_decoder = wandb.config.d2p_lang_embedding_when_decoder,

            feedforward_dim = wandb.config.d2p_feedforward_dim, 
            embedding_dim = wandb.config.d2p_embedding_dim,
            model_size = wandb.config.d2p_model_size,

            init_embedding = None,
            training_mode = 'encodeWithLangEmbedding',
            logger_prefix = 'd2p',
            
            encoder_takes_prev_decoder_out=False,
            prompt_mlp_with_one_hot_lang=False,
            gated_mlp_by_target_lang=False,
            all_lang_summary_only = True,

            use_xavier_init = wandb.config.use_xavier_init,
            lr = wandb.config.lr,
            warmup_epochs = wandb.config.warmup_epochs,
            max_epochs = wandb.config.max_epochs,
            
            beta1 = wandb.config.beta1,
            beta2 = wandb.config.beta2,
            eps = wandb.config.eps,
        )
    case ('p2d', 'GRU'):
        model = Seq2SeqRNN(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab, 
            
            num_encoder_layers = wandb.config.p2d_num_encoder_layers,
            dropout_p = wandb.config.p2d_dropout_p,
            use_vae_latent = wandb.config.p2d_use_vae_latent,
            inference_decode_max_length = wandb.config.p2d_inference_decode_max_length,
            use_bidirectional_encoder = wandb.config.p2d_use_bidirectional_encoder,
            decode_mode = wandb.config.p2d_decode_mode,
            beam_search_alpha = None if wandb.config.p2d_beam_search_alpha == "None" else wandb.config.p2d_beam_search_alpha,
            beam_size = None if wandb.config.p2d_beam_size == "None" else wandb.config.p2d_beam_size,
            lang_embedding_when_decoder = wandb.config.p2d_lang_embedding_when_decoder,
            prompt_mlp_with_one_hot_lang = wandb.config.p2d_prompt_mlp_with_one_hot_lang,
            gated_mlp_by_target_lang = wandb.config.p2d_gated_mlp_by_target_lang,

            feedforward_dim = wandb.config.p2d_feedforward_dim, 
            embedding_dim = wandb.config.p2d_embedding_dim,
            model_size = wandb.config.p2d_model_size,

            init_embedding = None,
            training_mode = 'encodeTokenSeqAppendTargetLangToken',
            logger_prefix = 'p2d',
            
            encoder_takes_prev_decoder_out=False,
            all_lang_summary_only  = wandb.config.p2d_all_lang_summary_only,

            use_xavier_init = wandb.config.use_xavier_init,
            lr = wandb.config.lr,
            warmup_epochs = wandb.config.warmup_epochs,
            max_epochs = wandb.config.max_epochs,
            
            beta1 = wandb.config.beta1,
            beta2 = wandb.config.beta2,
            eps = wandb.config.eps,
        )
    case ('d2p', 'Transformer'):
        model = EncoderDecoderTransformer(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab,
            num_encoder_layers = wandb.config.d2p_num_encoder_layers,
            num_decoder_layers = wandb.config.d2p_num_decoder_layers,
            embedding_dim = wandb.config.d2p_embedding_dim,
            nhead = wandb.config.d2p_nhead,
            feedforward_dim = wandb.config.d2p_feedforward_dim,
            dropout_p = wandb.config.d2p_dropout_p,
            max_len = wandb.config.d2p_max_len,
            logger_prefix = 'd2p',
            task = 'd2p',
            inference_decode_max_length = wandb.config.d2p_inference_decode_max_length,
            all_lang_summary_only = True,
            
            use_xavier_init = wandb.config.use_xavier_init,
            lr = wandb.config.lr,
            warmup_epochs = wandb.config.warmup_epochs,
            max_epochs = wandb.config.max_epochs,
            weight_decay = wandb.config.weight_decay,

            beta1 = wandb.config.beta1,
            beta2 = wandb.config.beta2,
            eps = wandb.config.eps,
        )
    case ('p2d', 'Transformer'):
        model = EncoderDecoderTransformer(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab,
            num_encoder_layers = wandb.config.p2d_num_encoder_layers,
            num_decoder_layers = wandb.config.p2d_num_decoder_layers,
            embedding_dim = wandb.config.p2d_embedding_dim,
            nhead = wandb.config.p2d_nhead,
            feedforward_dim = wandb.config.p2d_feedforward_dim,
            dropout_p = wandb.config.p2d_dropout_p,
            max_len = wandb.config.p2d_max_len,
            logger_prefix = 'p2d',
            task = 'p2d',
            inference_decode_max_length = wandb.config.p2d_inference_decode_max_length,
            all_lang_summary_only = wandb.config.p2d_all_lang_summary_only,
            
            use_xavier_init = wandb.config.use_xavier_init,
            lr = wandb.config.lr,
            warmup_epochs = wandb.config.warmup_epochs,
            max_epochs = wandb.config.max_epochs,
            weight_decay = wandb.config.weight_decay,
            
            beta1 = wandb.config.beta1,
            beta2 = wandb.config.beta2,
            eps = wandb.config.eps,
        )
    case ('p2d', 'JambuGRU'):
        assert wandb.config.beta1 == 0.9
        assert wandb.config.beta2 == 0.98
        assert wandb.config.eps == 1e-9
        model = JambuGRU(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab,
            emb_size = wandb.config.emb_size,
            hidden_size = wandb.config.hidden_size,
            num_layers = wandb.config.num_layers,
            dropout = wandb.config.dropout,
            bidirectional = wandb.config.bidirectional,
            lr = wandb.config.lr,
            logger_prefix = 'p2d',
            inference_decode_max_length = wandb.config.p2d_inference_decode_max_length,
            all_lang_summary_only = wandb.config.p2d_all_lang_summary_only,
            warmup_epochs = wandb.config.warmup_epochs,
            max_epochs = wandb.config.max_epochs,
        )
    case ('p2d', 'JambuTransformer'):
        assert wandb.config.beta1 == 0.9
        assert wandb.config.beta2 == 0.98
        assert wandb.config.eps == 1e-9
        model = JambuTransformer(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab,
            n_layers = wandb.config.n_layers,
            d_model = wandb.config.d_model,
            d_ff = wandb.config.d_ff,
            n_heads = wandb.config.n_heads,
            dropout = wandb.config.dropout,
            lr = wandb.config.lr,
            logger_prefix = 'p2d',
            inference_decode_max_length = wandb.config.p2d_inference_decode_max_length,
            all_lang_summary_only = wandb.config.p2d_all_lang_summary_only,
            warmup_epochs = wandb.config.warmup_epochs,
            max_epochs = wandb.config.max_epochs,
        )
    case _:
        raise ValueError
# endregion

callbacks: list[Callback] = []
if sweeping:
    callbacks.append(EarlyStopping(
        monitor=f"{submodel}/val/phoneme_edit_distance", mode="min",
        patience = 7 if not dev else 3,
    ))
if not sweeping:
    callbacks.append(ModelCheckpoint(
        monitor=f"{submodel}/val/phoneme_edit_distance",
        mode="min",
        save_top_k=1,
        verbose=dev,
    ))

# region === training ===
trainer = pl.Trainer(
    accelerator = 'cpu' if cpu else 'auto',
    devices = set_devices,
    max_epochs = wandb.config.max_epochs if not dev else 10,
    log_every_n_steps=10,
    num_sanity_val_steps=1, 
    fast_dev_run=False,
    logger=wandb_logger,
    check_val_every_n_epoch = wandb.config.check_val_every_n_epoch if not dev else 1000,
    callbacks=callbacks,
)

if wandbwatch:
    wandb_logger.watch(model)
trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
if not sweeping:
    trainer.test(model, ckpt_path="best", dataloaders=dm.test_dataloader())
# endregion