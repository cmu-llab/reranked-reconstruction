# Code in this file is adapted from the "Jambu: A historical linguistic database for South Asian languages" paper (Arora et al. 2023)
# Their repository is at https://github.com/moli-mandala/data
# We made modifications to:
# 1. Integrate the model into our reranking system that uses the PyTorch Lightning framework
# 2. Modified to work on our datasets and evaluation metrics
# 3. Batched inference
# 4. Match the scheduler of our other reflex prediction models

from collections import defaultdict
from heapq import heappop, heappush
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import log_softmax, pad
import math
import copy
import pytorch_lightning as pl
from tqdm import tqdm
from specialtokens import *
from prelude import *
from lib.vocab import Vocab
import models.utils as utils
from sacrebleu.metrics import BLEU, CHRF, TER
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
import transformers

bleu = BLEU()
chrf = CHRF()
ter = TER()

# region === helper code from the Arora et al. (2023), slightly modified to work with our dataset ===

def make_model(
    src_vocab, tgt_vocab, N=2, d_model=128, d_ff=1024, h=8, dropout=0.2
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = JambuTransformerEncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0, unsq=False):
        # NOTE: unsq should be true for transformer models
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            if unsq:
                self.trg_mask = self.make_std_mask(self.trg, pad_index)
            else:
                self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
    @staticmethod
    def make_std_mask(trg, pad):
        "Create a mask to hide padding and future words."
        trg_mask = (trg != pad).unsqueeze(-2)
        trg_mask = trg_mask & subsequent_mask(trg.size(-1)).type_as(
            trg_mask.data
        )
        return trg_mask
    
def greedy_decode(model, src, src_mask, src_lengths, max_len=100):
    """Greedily decode a sentence."""
    output = []
    attention_scores = []
    prod = 0

    memory = model.encode(src, src_mask, src_lengths)
    ys = torch.zeros(1, 1).fill_(BOS_IDX).type_as(src.data)

    for i in range(max_len):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        prod += _
        output.append(next_word.data.item())
        next_word = next_word.data[0]

        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if next_word == EOS_IDX:
            break
    
    for i in range(len(model.encoder.layers)):
        attention_scores.append([model.encoder.layers[i].self_attn.attn.cpu().numpy()])
    for i in range(len(model.decoder.layers)):
        attention_scores.append([model.decoder.layers[i].self_attn.attn.cpu().numpy()])
        attention_scores.append([model.decoder.layers[i].src_attn.attn.cpu().numpy()])

    output = np.array(output)

    # cut off everything starting from </s> 
    # (only when EOS provided)
    if EOS_IDX is not None:
        first_eos = np.where(output==EOS_IDX)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, attention_scores, prod

def get_predictions(model, batch: Batch, max_len: int, reverse_mapping: dict, 
    maxi=None, pr=False, beam=None # these three are not used
):
    """Greedy decode predictions from a batch."""
    length = len(batch.src_lengths)
    res = []
    pred_ids_acc = []
    if maxi is not None:
        length = min(maxi, length)
    for i in range(length):
        src = [reverse_mapping[x.item()] for x in batch.src[i] if x.item() != PAD_IDX]
        trg = [reverse_mapping[x.item()] for x in batch.trg[i] if x.item() != PAD_IDX]

        pred_ids, attns, probs = greedy_decode(
            model, batch.src[i].reshape(1, -1), batch.src_mask[i].reshape(1, -1),
            [batch.src_lengths[i]],
            max_len = max_len
        )
        pred_ids_acc.append(torch.tensor(pred_ids))
        pred = [reverse_mapping[x.item()] for x in pred_ids if x.item() != PAD_IDX]
        res.append([src, trg, pred])

        # print
        if pr:
            print(i)
            print(' '.join(src))
            print(' '.join(trg))
            print(' '.join(pred), f"({probs.exp().detach().item():.6%})")
            print()
    
    return res

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None, scheduler=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.scheduler = scheduler

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm
        
        return loss

def make_std_mask(trg, pad):
    "Create a mask to hide padding and fu-ture words."
    trg_mask = (trg != pad).unsqueeze(-2)
    trg_mask = trg_mask & subsequent_mask(trg.size(-1)).type_as(
        trg_mask.data
    )
    return trg_mask

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

# endregion
 
class JambuTransformer(pl.LightningModule):
    def __init__(self,
        ipa_vocab: Vocab,
        lang_vocab: Vocab, # dummy so that API calls work. not used for language embedding

        n_layers: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,

        lr: float,
        warmup_epochs: int,
        max_epochs: int,
        logger_prefix: str,
        inference_decode_max_length: int,
        all_lang_summary_only: bool,
    ):
        super().__init__()
        self.ipa_vocab = ipa_vocab
        self.lang_vocab = lang_vocab
        self.inference_decode_max_length = inference_decode_max_length
        emb_size = d_model # same as original implementation
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        vocab_size = len(ipa_vocab)
        self.wrapped_model = make_model(vocab_size, vocab_size, n_layers, d_model, d_ff, n_heads, dropout)
        
        self.logger_prefix = logger_prefix
        assert self.logger_prefix == 'p2d' # this model should only be used for reflex prediction
        self.all_lang_summary_only = all_lang_summary_only
        
        self.eval_step_outputs = []
        self.evaled_on_target_langs = set() # keep track of which target langs we've evaled on if we're using encodeTokenSeqAppendTargetLangToken

        self.criterion = LabelSmoothing(size=vocab_size, padding_idx=PAD_IDX, smoothing=0.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            self.warmup_epochs,
            self.max_epochs,
            lr_end=0.000001
        )
        self.scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        
        # reranking support
        self.possible_target_lang_langs = self.lang_vocab.to_indices(self.lang_vocab.daughter_langs) # list of ints, indices in lang_vocab.
        self.min_possible_target_lang_langs_idx = min(self.possible_target_lang_langs)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler_config
        }    
        
    def forward_on_batch(self, batch):
        _, (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), _ = batch
        N = prompted_p_tkns.shape[0]
        
        src = prompted_p_tkns
        trg = d_tkns
        
        logits = self.wrapped_model.forward(
            src,
            trg[:, :-1],
            (prompted_p_tkns != PAD_IDX).unsqueeze(-2),
            make_std_mask(trg[:, :-1], PAD_IDX),
            prompetd_p_lens,
            None, # not used
        )

        return logits
        
    def training_step(self, batch, batch_idx):
        _, (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), _ = batch
        
        N = prompted_p_tkns.shape[0]
        trg = d_tkns
        
        # 1 > set up
        loss_compute = SimpleLossCompute(self.wrapped_model.generator, self.criterion, self.optimizer, None)
        
        # 2 > run epoch

        logits = self.forward_on_batch(batch)

        loss = loss_compute(
            logits, 
            trg[:, 1:], 
            N
        )

        # 3 > compute loss?
        
        self.log(f"{self.logger_prefix}/train/loss", loss, prog_bar=True)
        self.log(f"{self.logger_prefix}/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx, prefix='test')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx, prefix='val')

    # our batched implementation of greedy decode to speed up inference
    def batched_greedy_decode(self, src, src_mask, src_lengths, max_len=100) -> Tensor:
        output = []
        attention_scores = []
        N = src.shape[0]
        reached_eos = torch.zeros((N, 1), dtype=torch.bool).to(self.device)

        memory = self.wrapped_model.encode(src, src_mask, src_lengths)
        ys = torch.zeros(N, 1).fill_(BOS_IDX).type_as(src.data)
        
        for i in range(max_len - 1):
            out = self.wrapped_model.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.wrapped_model.generator(out[:, -1])
            # prob (N V)
            _, next_word = torch.max(prob, dim=1)
            # next_word (N)
            output.append(next_word)
            
            pred_tkn = (reached_eos == 1) * PAD_IDX | (reached_eos == 0) * rearrange(next_word, "N -> N 1")
            reached_eos = reached_eos | (pred_tkn == EOS_IDX)

            ys = torch.cat([ys, pred_tkn], dim=1)
            
            if torch.all(reached_eos == 1):
                break

        predictions = ys # (N, Ldecoded)

        return predictions

    def shared_eval_step(self, batch, batch_idx, prefix: str):

        _, (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), _ = batch
        source_tokens, source_langs, target_tokens, target_langs = prompted_p_tkns, None, d_tkns, d_lang_langs
   
        src = prompted_p_tkns
        trg = d_tkns        
        N = prompted_p_tkns.shape[0]
        
        # 1 > get loss
        loss_compute = SimpleLossCompute(self.wrapped_model.generator, self.criterion, self.optimizer, None)
        logits = self.forward_on_batch(batch)
        loss = loss_compute(
            logits, 
            trg[:, 1:], 
            N
        )
        self.log(f"{self.logger_prefix}/{prefix}/loss", loss, on_step=False, on_epoch=True, batch_size=N)

        # 2 > adapt batch from our format into original format
        
        converted_batch = Batch((prompted_p_tkns, [prompted_p_tkns.shape[-1]] * N), (d_tkns, [d_tkns.shape[-1]] * N), pad_index=PAD_IDX, unsq=True)
                
        # 3 > get predictions
        
        # commented: Arora et al. (2023)'s unbatched implementation
        # we're using our batched implementation
        
        # test = [converted_batch]
        # batch_size = N
        # reverse_mapping = self.ipa_vocab.i2v
        
        # res = get_predictions(self.wrapped_model, converted_batch, self.inference_decode_max_length, reverse_mapping, maxi=N, pr=False, beam=None)
        # gold, pred = [[' '.join(x[1][1:-1]) for x in res]], [' '.join(x[2]) for x in res]
        # b, c, t = bleu.corpus_score(pred, gold), chrf.corpus_score(pred, gold), ter.corpus_score(pred, gold)
        # self.log(f"{self.logger_prefix}/{prefix}/blue", b.score, on_step=False, on_epoch=True, batch_size=N)
        # self.log(f"{self.logger_prefix}/{prefix}/chrf", c.score, on_step=False, on_epoch=True, batch_size=N)
        # self.log(f"{self.logger_prefix}/{prefix}/ter", t.score, on_step=False, on_epoch=True, batch_size=N)
        
        # # convert back predictions
        # _predictions = [torch.tensor([self.ipa_vocab.v2i[idx] for idx in x[2]]) for x in res]
        # predictions = pad_sequence(_predictions, batch_first=True, padding_value=PAD_IDX)

        predictions = self.batched_greedy_decode(
            converted_batch.src, 
            converted_batch.src_mask, 
            converted_batch.src_lengths, 
            max_len=self.inference_decode_max_length
        )
                
        # 4 > do our evaluations
                
        string_res = utils.mk_strings_from_forward(self, source_tokens, source_langs, target_tokens, target_langs, predictions, None)

        for string_res_dict in string_res:        
            self.evaled_on_target_langs.add(string_res_dict['target_lang'])
            self.eval_step_outputs.append(string_res_dict)
    
    def on_validation_epoch_end(self):
        return self.shared_eval_epoch_end('val')
    
    def on_test_epoch_end(self):
        return self.shared_eval_epoch_end('test')

    def shared_eval_epoch_end(self, prefix: str):
        self.evaled_on_target_langs.add(ALL_TARGET_LANGS_LABEL)
        
        metric_out = utils.calc_metrics_from_string_dicts(
            self,
            self.evaled_on_target_langs, 
            self.eval_step_outputs,
            self.all_lang_summary_only,
            prefix,
        )
        
        # reset stuff
        self.evaled_on_target_langs.clear()
        self.eval_step_outputs.clear()
                
        return metric_out



class JambuTransformerEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(JambuTransformerEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, src_lengths, tgt_lengths):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, src_lengths), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

