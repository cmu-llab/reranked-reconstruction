import torch
from torch import Tensor
from tqdm import tqdm
from specialtokens import *
from prelude import ALL_TARGET_LANGS_LABEL
import panphon.distance
import lingrex.reconstruct
import numpy as np
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from prelude import batch_t
from torch import nn
from einops import repeat, rearrange

# region === data ===

def mk_masks_for_transformer( 
    s, # (N, L)
    t, # (N, L)
    device,
):
    s_len = s.shape[1]
    t_len = t.shape[1]

    t_mask = nn.Transformer.generate_square_subsequent_mask(t_len).bool().to(device)
    s_mask = torch.zeros((s_len, s_len)).bool().to(device)

    s_pad_mask = (s == PAD_IDX).bool()
    t_pad_mask = (t == PAD_IDX).bool()

    return s_mask, t_mask, s_pad_mask, t_pad_mask

def unpack_batch_for_transformer(
    batch: batch_t, 
    device, 
    task: str, # 'd2p' | 'p2d',
    ipa_vocab, 
    lang_vocab,
    protolang,
):
    (d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s) = batch
    
    match task:
        case 'd2p':
            s_tkns, s_langs, s_indv_lens, t_tkns = d_cat_tkns, d_cat_langs, d_indv_lens, p_tkns
        case 'p2d':
            s_tkns, s_langs, s_indv_lens, t_tkns = prompted_p_tkns, None, None, d_tkns
        case _: raise ValueError
    
    
    t_tkns_in = t_tkns[:, :-1] # (N, Lt-1)
    t_tkns_out = t_tkns[:, 1:] # (N, Lt-1)
    
    s_mask, t_mask, s_pad_mask, t_pad_mask = mk_masks_for_transformer(s_tkns, t_tkns_in, device)
    
    N = s_tkns.shape[0]
    
    match task:
        case 'd2p':
            p_lang_ipa_ids = repeat((torch.LongTensor([ipa_vocab.get_idx(protolang)]).to(device)), '1 -> N 1', N=N)
            p_lang_lang_ids = repeat((torch.LongTensor([lang_vocab.get_idx(protolang)]).to(device)), '1 -> N 1', N=N)
            t_ipa_lang = p_lang_ipa_ids
            t_lang_lang = p_lang_lang_ids
        case 'p2d':
            t_ipa_lang = d_ipa_langs
            t_lang_lang = d_lang_langs
        case _: raise ValueError
    
    return N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask

# endregion

# region === evaluation ===

def mk_strings_from_forward(
    model, 
    source_tokens: Tensor, 
    source_langs: Tensor | None, 
    target_tokens: Tensor, 
    target_langs: Tensor | None, # note that target_langs uses language vocabulary
    predictions: Tensor, 
    processed_endnodes: Tensor | None
):
    
    res = []
    
    for i in range(source_tokens.shape[0]):

        match target_langs:
            case None:
                the_target_language = None
            case _:
                the_target_language = model.lang_vocab.to_tokens(target_langs[i].unsqueeze(0))[0]
        
        # with special tokens
        s_tkn_list_r = model.ipa_vocab.to_tokens(source_tokens[i], remove_special=False)
        s_lang_list_r = None if source_langs == None else model.lang_vocab.to_tokens(source_langs[i], remove_special=False)
        t_tkn_list_r = model.ipa_vocab.to_tokens(target_tokens[i], remove_special=False)
        t_hat_tkn_list_r = model.ipa_vocab.to_tokens(predictions[i], remove_special=False)
        
        s_str_r = ''.join(s_tkn_list_r)
        s_lang_str_r = None if s_lang_list_r == None else ''.join(s_lang_list_r)
        t_str_r = ''.join(t_tkn_list_r)
        t_hat_str_r = ''.join(t_hat_tkn_list_r)
        
        # without special tokens
        s_tkn_list = model.ipa_vocab.to_tokens(source_tokens[i], remove_special=True)
        s_lang_list = None if source_langs == None else model.lang_vocab.to_tokens(source_langs[i], remove_special=True)
        t_tkn_list = model.ipa_vocab.to_tokens(target_tokens[i], remove_special=True)
        t_hat_tkn_list = model.ipa_vocab.to_tokens(predictions[i], remove_special=True)
        
        s_str = ''.join(s_tkn_list)
        s_lang_str = None if s_lang_list == None else ''.join(s_lang_list)
        t_str = ''.join(t_tkn_list)
        t_hat_str = ''.join(t_hat_tkn_list) 
        
        t_rank_in_beam_search = None
        if processed_endnodes is not None:
            # figure out endnode position
            paocessed_endnodes_for_batch = processed_endnodes[i]
            ranked_endnodes_t_hat_str = [''.join(model.ipa_vocab.to_tokens(endnode_entry['endnode_seq'], remove_special=True)) for endnode_entry in paocessed_endnodes_for_batch]
            try:
                t_rank_in_beam_search = ranked_endnodes_t_hat_str.index(t_str)
            except ValueError: 
                t_rank_in_beam_search = None

        res.append({
            'target_lang': the_target_language,
            
            's_tkn_list_r': s_tkn_list_r,
            's_tkn_list': s_tkn_list,
            's_str_r': s_str_r,
            's_str': s_str,
            
            's_lang_list_r': s_lang_list_r,
            's_lang_list': s_lang_list,
            's_lang_str_r': s_lang_str_r,
            's_lang_str': s_lang_str,
            
            't_tkn_list_r': t_tkn_list_r,
            't_tkn_list': t_tkn_list,
            't_str_r': t_str_r,
            't_str': t_str,
            
            't_hat_tkn_list_r': t_hat_tkn_list_r,
            't_hat_tkn_list': t_hat_tkn_list,
            't_hat_str_r': t_hat_str_r,
            't_hat_str': t_hat_str,
            
            't_rank_in_beam_search': t_rank_in_beam_search,
        })
        
    return res

def calc_metrics_from_string_dicts(
    model, 
    evaled_on_target_langs: set, 
    eval_step_outputs: list[dict],
    all_lang_summary_only: bool,
    prefix: str,
):
    accumulators = {}
    for target_lang in evaled_on_target_langs:
        accumulators[target_lang] = {
            # all of these are accumulators
            'n_acc': 0, # num examples in eval epoch
            'char_edit_distance_acc': 0, 'phoneme_edit_distance_acc': 0,
            'n_correct_acc': 0,
            'total_t_phoneme_len_acc': 0, 'total_t_hat_phoneme_len_acc': 0,
            's_strs': [], 't_strs': [], 't_hat_strs': [],
            'phoneme_pairs': [], # list of tuples of (target phoneme list, predicted phoneme list)                
            't_rank_in_beam_search_acc': [], # list[int]
        }
    eval_samples_table_acc = {"s": [], "t": [], "t_hat": []}
    
    # accumulate
    for i, output in tqdm(enumerate(eval_step_outputs), desc="aggragating eval stats", leave=False):
        the_target_lang = output['target_lang']
        def accumulate(k: str, v):
            if isinstance(accumulators[ALL_TARGET_LANGS_LABEL][k], list):
                accumulators[ALL_TARGET_LANGS_LABEL][k].append(v)
                if the_target_lang != None: 
                    accumulators[the_target_lang][k].append(v)
            else:
                accumulators[ALL_TARGET_LANGS_LABEL][k] += v
                if the_target_lang != None: 
                    accumulators[the_target_lang][k] += v

        accumulate('n_acc', 1)
        
        accumulate('char_edit_distance_acc', get_edit_distance(output['t_str'], output['t_hat_str']))
        accumulate('phoneme_edit_distance_acc', get_edit_distance(output['t_tkn_list'], output['t_hat_tkn_list']))
        
        accumulate('s_strs', output['s_str'])
        accumulate('t_strs', output['t_str'])
        accumulate('t_hat_strs', output['t_hat_str'])
        
        accumulate('total_t_phoneme_len_acc', len(output['t_tkn_list']))
        accumulate('total_t_hat_phoneme_len_acc', len(output['t_hat_tkn_list']))
        if output['t_str'] == output['t_hat_str']:
            accumulate('n_correct_acc', 1)

        accumulate('phoneme_pairs', (output['t_tkn_list'], output['t_hat_tkn_list']))

        # sample decoded outputs
        # if i < 20:
        #     eval_samples_table_acc['s'].append(output['s_str'])
        #     eval_samples_table_acc['t'].append(output['t_str'])
        #     eval_samples_table_acc['t_hat'].append(output['t_hat_str'])
            
        # beam search stats
        if 'decode_mode' in dir(model): 
            match model.decode_mode:
                case 'beam_search':
                    if output['t_rank_in_beam_search'] != None:
                        accumulate('t_rank_in_beam_search_acc', output['t_rank_in_beam_search'])
                case _: 
                    pass
    
    # log samples to table
    # if self.current_epoch > 0 and isinstance(self.logger, WandbLogger):
    #     # create samples table if not exist
    #     if self.eval_samples_table == None:
    #         self.eval_samples_table = wandb.Table(columns=["source"], data=[[item] for item in eval_samples_table_acc['s']])
    #         self.eval_samples_table.add_column(name="target", data=eval_samples_table_acc['t'])
    #     self.eval_samples_table.add_column(name=f"epoch_{self.current_epoch}", data=eval_samples_table_acc['t_hat'])
    #     if self.enable_logging:
    #         self.logger.experiment.log({f'{self.logger_prefix}/val/samples': copy(self.val_samples_table)}) # copy hack to workaround https://github.com/wandb/wandb/issues/2981
    
    # summarise for each target lang (including all lang as target lang)
    for target_lang in tqdm(evaled_on_target_langs, desc="creating per language summary", leave=False):
        if all_lang_summary_only and target_lang != ALL_TARGET_LANGS_LABEL:
            continue
        n_acc = accumulators[target_lang]['n_acc']
        char_edit_distance_acc = accumulators[target_lang]['char_edit_distance_acc']
        phoneme_edit_distance_acc = accumulators[target_lang]['phoneme_edit_distance_acc']
        n_correct_acc = accumulators[target_lang]['n_correct_acc']
        total_t_phoneme_len_acc = accumulators[target_lang]['total_t_phoneme_len_acc']
        total_t_hat_phoneme_len_acc = accumulators[target_lang]['total_t_hat_phoneme_len_acc']
        s_strs = accumulators[target_lang]['s_strs']
        t_strs = accumulators[target_lang]['t_strs']
        t_hat_strs = accumulators[target_lang]['t_hat_strs']
        phoneme_pairs = accumulators[target_lang]['phoneme_pairs']
        t_rank_in_beam_search_acc = accumulators[target_lang]['t_rank_in_beam_search_acc']
                        
        accuracy = n_correct_acc / n_acc
        
        avg_char_edit_distance = char_edit_distance_acc / n_acc
        avg_phoneme_edit_distance = phoneme_edit_distance_acc / n_acc
        phoneme_error_rate = phoneme_edit_distance_acc / total_t_phoneme_len_acc
        
        avg_t_phoneme_len = total_t_phoneme_len_acc / n_acc
        avg_t_hat_phoneme_len = total_t_hat_phoneme_len_acc / n_acc
        
        feature_error_rate = get_feature_error_rate(t_hat_strs, t_strs)
        bcubed_f_score = get_bcubed_f_score(phoneme_pairs)
        
        should_record_beam_stats: bool = 'decode_mode' in dir(model) and model.decode_mode == 'beam_search' and len(t_rank_in_beam_search_acc) != 0
        if should_record_beam_stats: 
            avg_t_rank_in_beam_search = np.mean(t_rank_in_beam_search_acc)
            std_t_rank_in_beam_search = np.std(t_rank_in_beam_search_acc)
            target_in_beam = len(t_rank_in_beam_search_acc) / n_acc
                    
        # logger logging
        if target_lang == ALL_TARGET_LANGS_LABEL:
            res_dict = {
                f'{model.logger_prefix}/{prefix}/accuracy': accuracy,
                
                f'{model.logger_prefix}/{prefix}/char_edit_distance': avg_char_edit_distance,
                f'{model.logger_prefix}/{prefix}/phoneme_edit_distance': avg_phoneme_edit_distance,
                f'{model.logger_prefix}/{prefix}/phoneme_error_rate': phoneme_error_rate,
                f'{model.logger_prefix}/{prefix}/feature_error_rate': feature_error_rate,
                f'{model.logger_prefix}/{prefix}/bcubed_f_score': bcubed_f_score,
                
                f'{model.logger_prefix}/{prefix}/avg_target_phoneme_len': avg_t_phoneme_len,
                f'{model.logger_prefix}/{prefix}/avg_prediction_phoneme_len': avg_t_hat_phoneme_len,
            }
            # add beam search stats, if any
            if should_record_beam_stats:                    
                res_dict[f'{model.logger_prefix}/{prefix}/avg_t_rank_in_beam_search'] = avg_t_rank_in_beam_search
                res_dict[f'{model.logger_prefix}/{prefix}/std_t_rank_in_beam_search'] = std_t_rank_in_beam_search
                res_dict[f'{model.logger_prefix}/{prefix}/target_in_beam'] = target_in_beam
            
            try:
                model.log_dict(res_dict)
            except MisconfigurationException: pass
        elif len(evaled_on_target_langs) > 2: # if more than one target lang (set includes one special all target as target lang)
            res_dict = {
                f'{model.logger_prefix}/{prefix}/{target_lang}/accuracy': accuracy,
                
                f'{model.logger_prefix}/{prefix}/{target_lang}/char_edit_distance': avg_char_edit_distance,
                f'{model.logger_prefix}/{prefix}/{target_lang}/phoneme_edit_distance': avg_phoneme_edit_distance,
                f'{model.logger_prefix}/{prefix}/{target_lang}/phoneme_error_rate': phoneme_error_rate,
                f'{model.logger_prefix}/{prefix}/{target_lang}/feature_error_rate': feature_error_rate,
                f'{model.logger_prefix}/{prefix}/{target_lang}/bcubed_f_score': bcubed_f_score,
            }
            # add beam search stats, if any
            if should_record_beam_stats:
                res_dict[f'{model.logger_prefix}/{prefix}/{target_lang}/avg_t_rank_in_beam_search'] = avg_t_rank_in_beam_search
                res_dict[f'{model.logger_prefix}/{prefix}/{target_lang}/std_t_rank_in_beam_search'] = std_t_rank_in_beam_search
                res_dict[f'{model.logger_prefix}/{prefix}/{target_lang}/target_in_beam'] = target_in_beam

            try:
                model.log_dict(res_dict)
            except MisconfigurationException: pass
        else:
            pass       

# source: https://github.com/shauli-ravfogel/Latin_reconstruction, obtained from https://github.com/cmu-llab/meloni-2021-reimplementation/blob/main/main.py
def get_edit_distance(s1, s2):
    if type(s1) == str and type(s2) == str:
        s1 = s1.replace("<", "").replace(">", "")
        s2 = s2.replace("<", "").replace(">", "")

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_feature_error_rate(ss1: list[str], ss2: list[str]) -> float:
    dist = panphon.distance.Distance()
    return dist.feature_error_rate(ss1, ss2)

def get_bcubed_f_score(phoneme_pairs) -> float:
    return lingrex.reconstruct.eval_by_bcubes(phoneme_pairs)

# endregion

# region === loss ===

cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# expecting:
# logits: (N, C, d1, ..., dk) containing probabilities
# target_tokens: (N, d1, ..., dk) containing class indices
def calc_cross_entropy_loss(logits: Tensor, target_tokens: Tensor) -> Tensor:
    return cross_entropy_loss_fn(logits, target_tokens)

# endregion

# region === VAE (based on Cui et al. (2022)) ===

def vae_reparametrise(mu, logvar, sampling=True) -> Tensor:
    if sampling:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
    else:
        return mu

def vae_calc_kl_loss(mu, logvar, reduction='mean') -> Tensor:
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()
    return kl_loss

def calc_vae_losses(recon_logits, target_tokens, mu, logvar) -> tuple[Tensor, Tensor]:
    recon_loss = calc_cross_entropy_loss(recon_logits, target_tokens)
    kl_loss = vae_calc_kl_loss(mu, logvar)
    return recon_loss, kl_loss

# endregion