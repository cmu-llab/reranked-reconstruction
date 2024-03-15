# Code in this file is adapted from the "Jambu: A historical linguistic database for South Asian languages" paper (Arora et al. 2023)
# Their repository is at https://github.com/moli-mandala/data
# We made modifications to:
# 1. Integrate the model into our reranking system that uses the PyTorch Lightning framework
# 2. Modified to work on our datasets and evaluation metrics
# 3. Batched inference
# 4. Match the scheduler of our other reflex prediction models

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import log_softmax, pad
import math
import copy

from tqdm import tqdm
import models.utils as utils

from specialtokens import *
from prelude import *
from lib.vocab import Vocab
import models.utils as utils
from sacrebleu.metrics import BLEU, CHRF, TER
import transformers

bleu = BLEU()
chrf = CHRF()
ter = TER()

# region === helper code from the Arora et al. (2023), slightly modified to work with our dataset ===

def make_gru_model(src_vocab, tgt_vocab, emb_size, hidden_size, num_layers, dropout, bidirectional):
    "Helper: Construct a model from hyperparameters."
    attention = GRUBahdanauAttention(hidden_size, bidirectional=bidirectional)

    model = GRUEncoderDecoder(
        GRUEncoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional),
        GRUDecoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        GRUGenerator(hidden_size, tgt_vocab))
    
    return model

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    # Note: unsq should be true for transformer model
    def __init__(self, src, trg, pad_index=0, unsq=False):
        
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

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(BOS_IDX).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
            encoder_hidden, encoder_final, src_mask,
            prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        prod += _
        next_word = next_word.data.item()
        output.append(next_word)

        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
        if next_word == EOS_IDX:
            break
        
    output = np.array(output)

    # cut off everything starting from </s> 
    # (only when EOS provided)
    if EOS_IDX is not None:
        first_eos = np.where(output==EOS_IDX)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, attention_scores, prod

#[deprecated]. We use a batched version within the JambuGRU class
def get_predictions(model, batch: Batch, reverse_mapping: dict, maxi=None, pr=False, beam=None):
    """Greedy decode predictions from a batch."""
    length = len(batch.src_lengths)
    res = []
    if maxi is not None:
        length = min(maxi, length)
    for i in range(length):
        src = [reverse_mapping[x.item()] for x in batch.src[i] if x.item() != PAD_IDX]
        trg = [reverse_mapping[x.item()] for x in batch.trg[i] if x.item() != PAD_IDX]

        pred, attns, probs = greedy_decode(
            model, batch.src[i].reshape(1, -1), batch.src_mask[i].reshape(1, -1),
            [batch.src_lengths[i]],
            max_len = 15
        )
        pred = [reverse_mapping[x.item()] for x in pred if x.item() != PAD_IDX]
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

# endregion

# region === Arora et al. (2023)'s GRU model wrapped inside a module for modular testing ===
# we also implemented batched greedy decode to speed up inference

class JambuGRU(pl.LightningModule):
    def __init__(
        self, 
        ipa_vocab: Vocab,
        lang_vocab: Vocab, # dummy so that API calls work. not used for language embedding
        
        emb_size: int, 
        hidden_size: int, 
        num_layers: int, 
        dropout: float,
        bidirectional: bool,
        
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
        vocab_size = len(ipa_vocab)
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        self.wrapped_model = make_gru_model(vocab_size, vocab_size, emb_size, hidden_size, num_layers, dropout, bidirectional)
        
        self.criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        # Note:
        # - Adam hyperparams are from original code
        # - we used the same scheduler for all models for fair comparison
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

        self.logger_prefix = logger_prefix
        assert self.logger_prefix == 'p2d' # this model should only be used for reflex prediction
        self.all_lang_summary_only = all_lang_summary_only
        
        self.eval_step_outputs = []
        self.evaled_on_target_langs = set() # keep track of which target langs we've evaled on if we're using encodeTokenSeqAppendTargetLangToken

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler_config
        }    

    def greedy_decode_batched(self, srcs, src_masks, src_lengths, max_len):
        '''batched greedy decode w/ teacher forcing'''
        with torch.no_grad():
            enc_out, enc_hns = self.wrapped_model.encode(srcs, src_masks, src_lengths)

        '''Give BOS as starting token'''
        prev_tkn = repeat(torch.tensor([BOS_IDX]), "h -> b h", b = srcs.size(0)).to(self.device)        # prev_tkn: (batch_size, 1)
        trg_mask = torch.ones_like(prev_tkn).to(self.device)                                            # trg_mask: (batch_size, 1)

        dec_hn = None
        N = srcs.size(0)
        pred_res_lis = []
        reached_eos = torch.zeros((N, 1), dtype=torch.bool).to(self.device)

        '''According to Arora et al. (2023)'s unbatched greedy decode, each predicted character will be fed into the decoder as prev token to predict the next char'''
        for i in range(max_len):
            dec_out, dec_hn, pred_logits = self.wrapped_model.decode(enc_out, enc_hns, src_masks, prev_tkn, trg_mask, dec_hn)     # pred_logits: (batch_size, 1, hidden_size)
            prob = self.wrapped_model.generator(pred_logits)      # prob: (batch_size, 1, vocab_size)

            pred_tkn = torch.argmax(prob, dim = 2)  # pred_tkn: (batch_size, 1)
            
            pred_tkn = (reached_eos == 1) * PAD_IDX | (reached_eos == 0) * pred_tkn
            reached_eos = reached_eos | (pred_tkn == EOS_IDX)
            pred_res_lis.append(pred_tkn)

            if torch.all(reached_eos == 1):
                break

            prev_tkn = pred_tkn

        pred_res = torch.stack(pred_res_lis, dim = 1).squeeze(2)       # pred_res: (batch_size, trg_seq_len)
        return pred_res

    def batched_get_predictions(self, batch: Batch):
        """Greedy decode predictions from a batch."""
        res = []

        pred = self.greedy_decode_batched(
            batch.src, 
            batch.src_mask,
            batch.src_lengths,
            max_len = self.inference_decode_max_length,
        )
        
        for i in range(batch.trg.size(0)):
            src = [self.ipa_vocab.i2v[x.item()] for x in batch.src[i]]
            pred_tkn = [self.ipa_vocab.i2v[x.item()] for x in pred[i]]
            trg = [self.ipa_vocab.i2v[x.item()] for x in batch.trg[i]]
            res.append((src, trg, pred_tkn))
            
        return pred, res 
    
    def forward_on_batch(self, batch):
        _, (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), _ = batch
        
        src = prompted_p_tkns
        trg = d_tkns

        pre_output = self.wrapped_model.forward(
            src,
            trg[:, :-1],
            (prompted_p_tkns != PAD_IDX).unsqueeze(-2),
            make_std_mask(trg[:, :-1], PAD_IDX),
            prompetd_p_lens.squeeze(-1),
            None, # not used
        )
        
        out, _, logits = pre_output

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

        # 3 > compute loss
        self.log(f"{self.logger_prefix}/train/loss", loss, prog_bar=True)
        self.log(f"{self.logger_prefix}/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx, prefix='test')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx, prefix='val')

    def shared_eval_step(self, batch, _batch_idx, prefix: str):
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
        
        converted_batch = Batch((prompted_p_tkns, [prompted_p_tkns.shape[-1]] * N), (d_tkns, [d_tkns.shape[-1]] * N), pad_index=PAD_IDX, unsq=False)
                
        # 3 > get predictions         
        
        predictions, res = self.batched_get_predictions(converted_batch)
        
        gold, pred = [[' '.join(x[1][1:-1]) for x in res]], [' '.join(x[2]) for x in res]
        b, c, t = bleu.corpus_score(pred, gold), chrf.corpus_score(pred, gold), ter.corpus_score(pred, gold)
        self.log(f"{self.logger_prefix}/{prefix}/blue", b.score, on_step=False, on_epoch=True, batch_size=N)
        self.log(f"{self.logger_prefix}/{prefix}/chrf", c.score, on_step=False, on_epoch=True, batch_size=N)
        self.log(f"{self.logger_prefix}/{prefix}/ter", t.score, on_step=False, on_epoch=True, batch_size=N)

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

# endregion


# region === Below are code from Arora et al. (2023), with some variable renamings and annotations ===

class GRUEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(GRUEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)

class GRUBahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None, bidirectional=False):
        super(GRUBahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = (2 * hidden_size if bidirectional else hidden_size) if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

class GRUEncoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(GRUEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        # mask = repeat(torch.logical_not(mask),"b l -> b l e", e = x.shape[2])
        # x -= torch.where(mask, 1e10, 0.)
        if (isinstance(lengths, torch.Tensor)):
            lengths = lengths.to('cpu')
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)     # (batch_size, longest_proto_seq_len, embedding_dim)
        output, hn = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)       # (batch_size, longest_proto_seq_len, gru_hidden_dim * D)
        
        # we need to manually concatenate the final states for both directions
        if self.bidirectional:
            fwd_final = hn[0:hn.size(0):2]                  
            bwd_final = hn[1:hn.size(0):2]
            hn = torch.cat([fwd_final, bwd_final], dim=2)                           
            # hn was obtained by :torch.cat([fwd_final, bwd_final], dim=2), which makes hn (num_layers, batch_size, gru_hidden_dim*2)
            # But according to https://pytorch.org/docs/stable/generated/torch.nn.GRU.html , hn should be (num_layers*D, batch_size, gru_hidden_dim)???
            
        return output, hn

class GRUDecoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True, bidirectional=False):
        super(GRUDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.bidirectional = bidirectional
        bridge = bidirectional
                 
        self.rnn = nn.GRU(emb_size + (2*hidden_size if bidirectional else hidden_size), hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear((2*hidden_size if bidirectional else hidden_size), hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + (2*hidden_size if bidirectional else hidden_size) + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        if self.bridge:
            return torch.tanh(self.bridge(encoder_final))
        else:
            return encoder_final

class GRUGenerator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(GRUGenerator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# endregion