# rerankers and score interpolation functions
# our experiments use BatchedCorrectRateReranker and BatchedLinearRescorer

import torch
from torch import Tensor
import pytorch_lightning as pl
from einops import repeat, rearrange
from models.encoderDecoderRNN import Seq2SeqRNN
from lib.tensor_utils import num_sequences_equal
import statistics
import models.utils as utils
from prelude import *
from specialtokens import *
from lib.tensor_utils import collate_endnodes_to_tensors, padded_stack, sequences_equal, sort_by_permutation_3d
from models.encoderDecoderTransformer import EncoderDecoderTransformer
from models.jambuTransformer import JambuTransformer
from models.jambuTransformer import Batch as jambuTransformerBatch


class Reranker():
    def __init__(self) -> None:
        pass
    
    def __call__(self, endnode_seq: Tensor):
        raise NotImplementedError

class BatchedReranker(Reranker):
    def __init__(self) -> None:
        pass
    
    def __call__(self, 
        N,
        Nc,
        Nd,
        target_ipa_langs_s,
        target_lang_langs_s,
        broadcasted_collated_candidates,
        broadcasted_valid_daughters_mask,
        broadcasted_daughter_seqs_s,
    ):
        raise NotImplementedError

class BatchedGreedyBasedRerankerBase(BatchedReranker):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN | EncoderDecoderTransformer, 
    ):
        super().__init__()
        self.p2d_submodel = p2d_submodel

    def predict_daughters_from_candidates(self, 
        N: int, # batch size
        Nc: int, # number of candidates
        Nd: int, # number of daughters
        target_ipa_langs_s: Tensor, # (N, Nd, 1)
        target_lang_langs_s: Tensor, # (N, Nd, 1)
        broadcasted_collated_candidates: Tensor, # (N, Nc, Nd, Lc)
        broadcasted_valid_daughters_mask: Tensor # (N, Nc, Nd, 1)
    ) -> Tensor:
        
        self.p2d_submodel.to(target_ipa_langs_s.device)
        # collated_candidate (N, Nc, Nd, Lc)
        # candidate_reranker_score (N, Nc, Nd, 1)

        target_prompted_predicted_proto_s = torch.cat((
            repeat(target_ipa_langs_s, 'N Nd 1 -> N Nc Nd 1', Nc=Nc), 
            broadcasted_collated_candidates
        ), dim=-1) # (N, Nc, Nd, Lc+1)
        
        target_prompted_predicted_proto_lens_s = ((torch.sum(target_prompted_predicted_proto_s != PAD_IDX, dim=-1).unsqueeze(-1) * broadcasted_valid_daughters_mask.float()) + (1 - broadcasted_valid_daughters_mask.float())).long() # (N, Nc, Nd, 1)
        
        broadcasted_target_lang_langs_s = repeat(target_lang_langs_s, 'N Nd 1 -> N Nc Nd 1', Nc=Nc) # (N, Nc, Nd, 1)
        
        broadcasted_target_lang_langs_s_with_dummy_for_pad = None if broadcasted_target_lang_langs_s == None else broadcasted_target_lang_langs_s + ((1 * (~ broadcasted_valid_daughters_mask)) * self.p2d_submodel.min_possible_target_lang_langs_idx) # (N, Nc, Nd, 1)

        match self.p2d_submodel:
            case Seq2SeqRNN():
                predicted_daughters_s = rearrange(self.p2d_submodel.greedy_decode(
                    rearrange(target_prompted_predicted_proto_s, 'N Nc Nd L -> (N Nc Nd) L'), 
                    None, 
                    rearrange(target_prompted_predicted_proto_lens_s, 'N Nc Nd 1 -> (N Nc Nd) 1'), 
                    rearrange(broadcasted_target_lang_langs_s_with_dummy_for_pad, 'N Nc Nd 1 -> (N Nc Nd) 1'),
                ), '(N Nc Nd) L -> N Nc Nd L', N=N, Nc=Nc, Nd=Nd) # (N, Nc, Nd, Ld)
            case EncoderDecoderTransformer():
                s_len = target_prompted_predicted_proto_s.shape[3]
                s_tkns = rearrange(target_prompted_predicted_proto_s, 'N Nc Nd L -> (N Nc Nd) L')
                s_lens = None # individual lens are none for p2d models
                s_langs = None
                s_mask = torch.zeros((s_len, s_len)).bool().to(s_tkns.device)
                s_pad_mask = rearrange((target_prompted_predicted_proto_s == PAD_IDX).bool(), 'N Nc Nd L -> (N Nc Nd) L')
                decode_max_len = self.p2d_submodel.inference_decode_max_length
                
                predicted_daughters_s = rearrange(self.p2d_submodel.greedy_decode(
                    s_tkns, 
                    s_lens, 
                    s_langs, 
                    s_mask, 
                    s_pad_mask, 
                    decode_max_len,
                ), '(N Nc Nd) L -> N Nc Nd L', N=N, Nc=Nc, Nd=Nd) # (N, Nc, Nd, Ld)
            case JambuTransformer():
                s_tkns = rearrange(target_prompted_predicted_proto_s, 'N Nc Nd L -> (N Nc Nd) L')
                dummy_d_tkns = rearrange(torch.ones(N, Nc, Nd, 2), 'N Nc Nd L -> (N Nc Nd) L')
                
                converted_batch = jambuTransformerBatch(
                    (
                        s_tkns, 
                        [s_tkns.shape[-1]] * N
                    ), 
                    (
                        dummy_d_tkns, 
                        [dummy_d_tkns.shape[-1]] * N
                    ), 
                    pad_index=PAD_IDX, 
                    unsq=True
                )

                predicted_daughters_s = rearrange(self.p2d_submodel.batched_greedy_decode(
                    converted_batch.src, 
                    converted_batch.src_mask, 
                    converted_batch.src_lengths, 
                    max_len=self.p2d_submodel.inference_decode_max_length
                ), '(N Nc Nd) L -> N Nc Nd L', N=N, Nc=Nc, Nd=Nd) # (N, Nc, Nd, Ld)
            case _:
                raise NotImplementedError
        
        
        return predicted_daughters_s # (N, Nc, Nd, Ld)

class BatchedRandomReranker(BatchedGreedyBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__(p2d_submodel)

    def __call__(self,
        N,
        Nc,
        Nd,
        target_ipa_langs_s,
        target_lang_langs_s,
        broadcasted_collated_candidates,
        broadcasted_valid_daughters_mask,
        broadcasted_daughter_seqs_s,
    ):
        return torch.randn((N, Nc, 1)).to(broadcasted_collated_candidates.device)

class BatchedCorrectRateReranker(BatchedGreedyBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__(p2d_submodel)

    def __call__(self,
        N,
        Nc,
        Nd,
        target_ipa_langs_s,
        target_lang_langs_s,
        broadcasted_collated_candidates,
        broadcasted_valid_daughters_mask,
        broadcasted_daughter_seqs_s,
    ):
        predicted_daughters_s = self.predict_daughters_from_candidates( N, Nc, Nd, target_ipa_langs_s, target_lang_langs_s, broadcasted_collated_candidates, broadcasted_valid_daughters_mask) 
        # predicted_daughters_s (N, Nc, Nd, Ld)
        # recall broadcasted_daughter_seqs_s (N, Nc, Nd, Ld)
        
        correct_daughter_unmasked = rearrange(sequences_equal(
            rearrange(predicted_daughters_s, 'N Nc Nd L -> (N Nc Nd) L'), 
            rearrange(broadcasted_daughter_seqs_s, 'N Nc Nd L -> (N Nc Nd) L'),
        ), '(N Nc Nd) -> N Nc Nd 1', N=N, Nc=Nc, Nd=Nd)
        correct_daughter_masked = correct_daughter_unmasked & broadcasted_valid_daughters_mask # (N, Nc, Nd, 1)
        daughter_correct_rates = torch.sum(correct_daughter_masked.float(), dim=-2) / torch.sum(broadcasted_valid_daughters_mask.float(), dim=-2) # (N, Nc, 1)
        
        reranker_scores = daughter_correct_rates
        return reranker_scores # (N, Nc, 1)

class BatchedCharEditDistanceReranker(BatchedGreedyBasedRerankerBase):
    pass # not implemented

class BatchedPhonemeEditDistanceReranker(BatchedGreedyBasedRerankerBase):
    pass # not implemented

# returns a random score between 0 and 1, disregarding the input. This is effectively a baseline where the entries are reranked randomly
class RandomScoreReranker(Reranker):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        return torch.rand(1).item()    

# base reranker class for rerankers that use the greedy search
class GreedyBasedRerankerBase(Reranker):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__()
        self.p2d_submodel = p2d_submodel
        
    def interpret_beam_search_results(self, greedy_predicted_daughters, daughter_seqs):        
        raise NotImplementedError

    def __call__(self, 
        endnode_seq: Tensor, # (Lp)
        
        daughter_seqs: Tensor, # (Nd, Ld)
        target_ipa_langs: Tensor, # (Nd, 1)
        target_lang_langs: Tensor | None, # (Nd, 1)
    ):
        num_daughters: int = daughter_seqs.shape[0]
        target_prompted_predicted_proto = torch.cat((target_ipa_langs, repeat(endnode_seq, 'L -> Nd L', Nd=num_daughters)), dim=-1)
        
        target_prompted_predicted_proto_lens = repeat(torch.LongTensor([target_prompted_predicted_proto.shape[1]]), '1 -> Nd 1', Nd=num_daughters)
        
        predicted_daughters = self.p2d_submodel.greedy_decode(target_prompted_predicted_proto, None, target_prompted_predicted_proto_lens, target_lang_langs)
        
        return self.interpret_beam_search_results(predicted_daughters, daughter_seqs)

# rerank based on the correct rate of the daughter sequences given predicted proto
# score between 0 and 1
class CorrectRateReranker(GreedyBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__(p2d_submodel)

    def interpret_beam_search_results(self, greedy_predicted_daughters, daughter_seqs):        
        n_predicted_daughters = greedy_predicted_daughters.shape[0]
        n_predicted_daughters_correct = num_sequences_equal(greedy_predicted_daughters, daughter_seqs)
        daughter_correct_rate = n_predicted_daughters_correct / n_predicted_daughters
        return daughter_correct_rate                

# rerank based on the edit distance predicted daughters and gold daughters given predicted proto
# score between 0 and 1
class CharEditDistanceReranker(GreedyBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__(p2d_submodel)

    def interpret_beam_search_results(self, 
            greedy_predicted_daughters, # (Nd, Ld)
            daughter_seqs, # (Nd, Ld)
        ):        
        n_predicted_daughters = greedy_predicted_daughters.shape[0]

        seq2str = lambda seq: ''.join(self.p2d_submodel.ipa_vocab.to_tokens(seq, remove_special=True))
        get_edit_dist = lambda twoseqs: utils.get_edit_distance(*twoseqs)
        
        greedy_predicted_daughters_strs = map(seq2str, greedy_predicted_daughters) 
        daughter_seqs_strs = map(seq2str, daughter_seqs) 
        
        total_char_ed = sum(map(get_edit_dist, zip(daughter_seqs_strs, greedy_predicted_daughters_strs)))

        return - (total_char_ed / n_predicted_daughters)

class PhonemeEditDistanceReranker(GreedyBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__(p2d_submodel)

    def interpret_beam_search_results(self, 
            greedy_predicted_daughters, # (Nd, Ld)
            daughter_seqs, # (Nd, Ld)
        ):        
        n_predicted_daughters = greedy_predicted_daughters.shape[0]

        seq2phonemelist = lambda seq: self.p2d_submodel.ipa_vocab.to_tokens(seq, remove_special=True)
        get_edit_dist = lambda twoseqs: utils.get_edit_distance(*twoseqs)
        
        greedy_predicted_daughters_strs = map(seq2phonemelist, greedy_predicted_daughters) 
        daughter_seqs_strs = map(seq2phonemelist, daughter_seqs) 
        
        total_phoneme_ed = sum(map(get_edit_dist, zip(daughter_seqs_strs, greedy_predicted_daughters_strs)))
        
        return - (total_phoneme_ed / n_predicted_daughters)

class HybridGreedyBasedReranker(GreedyBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
    ):
        super().__init__(p2d_submodel)
        self.char_ed_reranker = CharEditDistanceReranker(p2d_submodel)
        self.phoneme_ed_reranker = PhonemeEditDistanceReranker(p2d_submodel)
        self.correct_rate_reranker = CorrectRateReranker(p2d_submodel)

    def interpret_beam_search_results(self, 
            greedy_predicted_daughters, # (Nd, Ld)
            daughter_seqs, # (Nd, Ld)
        ):
        char_ed_score = self.char_ed_reranker.interpret_beam_search_results(greedy_predicted_daughters, daughter_seqs) 
        phoneme_ed_score = self.phoneme_ed_reranker.interpret_beam_search_results(greedy_predicted_daughters, daughter_seqs)
        correct_rate_score = self.correct_rate_reranker.interpret_beam_search_results(greedy_predicted_daughters, daughter_seqs)
        
        return 0.2 * char_ed_score + 0.2 * phoneme_ed_score + 0.6 * correct_rate_score
        
# base reranker class for rerankers that use the beam search
class BeamBasedRerankerBase(Reranker):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
        beam_size: int,
    ):
        super().__init__()
        self.p2d_submodel = p2d_submodel
        self.beam_size = beam_size
        
    def interpret_beam_search_results(self, beam_search_res, daughter_seqs):
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = beam_search_res
        
        raise NotImplementedError

    def __call__(self, 
        endnode_seq: Tensor, # (Lp)
        
        daughter_seqs: Tensor, # (Nd, Ld)
        target_ipa_langs: Tensor, # (Nd, 1)
        target_lang_langs: Tensor | None, # (Nd, 1)
    ):
        num_daughters: int = daughter_seqs.shape[0]
        target_prompted_predicted_proto = torch.cat((target_ipa_langs, repeat(endnode_seq, 'L -> Nd L', Nd=num_daughters)), dim=-1)
        
        target_prompted_predicted_proto_lens = repeat(torch.LongTensor([target_prompted_predicted_proto.shape[1]]), '1 -> Nd 1', Nd=num_daughters)
        
        beam_search_res = self.p2d_submodel.beam_search_decode(target_prompted_predicted_proto, None, target_prompted_predicted_proto_lens, target_lang_langs, self.beam_size)
        
        return self.interpret_beam_search_results(beam_search_res, daughter_seqs)

class BeamPositionReranker(BeamBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
        beam_size: int,
        off_beam_penalty_multiplier: float,
    ):
        super().__init__(p2d_submodel, beam_size)
        self.off_beam_penalty = off_beam_penalty_multiplier * beam_size
        
    def interpret_beam_search_results(self, beam_search_res, daughter_seqs):
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = beam_search_res

        n_daughters = best_seqs_padded.shape[0]
        t_ranks = []
        for i in range(n_daughters):
            t_str = ''.join(self.p2d_submodel.ipa_vocab.to_tokens(daughter_seqs[i], remove_special=True))
            endnodes_for_daughter = processed_endnodes[i]
            ranked_endnodes_t_hat_str = [''.join(self.p2d_submodel.ipa_vocab.to_tokens(endnode_entry['endnode_seq'], remove_special=True)) for endnode_entry in endnodes_for_daughter]
            try:
                t_rank = ranked_endnodes_t_hat_str.index(t_str)
            except ValueError: 
                t_rank = self.beam_size + self.off_beam_penalty
            t_ranks.append(t_rank)
            
        mean_t_rank = sum(t_ranks) / len(t_ranks)

        return - (mean_t_rank / self.beam_size)

# rerank by the normalised log prob of the correct daughter sequence
class BeamLogProbReranker(BeamBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
        beam_size: int,
        off_beam_penalty_multiplier: float, # multiplies the lowest log prob in beam search res if not in beam
    ):
        super().__init__(p2d_submodel, beam_size)
        self.off_beam_penalty = off_beam_penalty_multiplier * beam_size
        
    def interpret_beam_search_results(self, beam_search_res, daughter_seqs):
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = beam_search_res

        n_daughters = best_seqs_padded.shape[0]
        norm_log_probs = []
        
        for i in range(n_daughters):
            t_str = ''.join(self.p2d_submodel.ipa_vocab.to_tokens(daughter_seqs[i], remove_special=True))
            endnodes_for_daughter = processed_endnodes[i]
            ranked_endnodes_t_hat_str = [''.join(self.p2d_submodel.ipa_vocab.to_tokens(endnode_entry['endnode_seq'], remove_special=True)) for endnode_entry in endnodes_for_daughter]
            try:
                t_rank = ranked_endnodes_t_hat_str.index(t_str)
                norm_log_prob = processed_endnodes[i][t_rank]['normalized_log_prob']
            except ValueError: 
                norm_log_prob = processed_endnodes[i][-1]['normalized_log_prob'] * self.off_beam_penalty
                
            norm_log_probs.append(norm_log_prob)

        mean_norm_log_prob = sum(norm_log_probs) / len(norm_log_probs)
        

        return mean_norm_log_prob

# rerank by how far the normalised log prob of the correct daughter sequence is away from top of the beam, harmonic mean
class BeamDeltaLogProbReranker(BeamBasedRerankerBase):
    def __init__(self,
        p2d_submodel: Seq2SeqRNN, 
        beam_size: int,
        off_beam_penalty_multiplier: float, # multiplies the delta of lowest log prob in beam search res if not in beam
    ):
        super().__init__(p2d_submodel, beam_size)
        self.off_beam_penalty = off_beam_penalty_multiplier * beam_size
        
    def interpret_beam_search_results(self, beam_search_res, daughter_seqs):
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = beam_search_res

        n_daughters = best_seqs_padded.shape[0]
        norm_log_prob_deltas = []
        
        for i in range(n_daughters):
            t_str = ''.join(self.p2d_submodel.ipa_vocab.to_tokens(daughter_seqs[i], remove_special=True))
            endnodes_for_daughter = processed_endnodes[i]
            ranked_endnodes_t_hat_str = [''.join(self.p2d_submodel.ipa_vocab.to_tokens(endnode_entry['endnode_seq'], remove_special=True)) for endnode_entry in endnodes_for_daughter]
            
            beam_top_log_prob = processed_endnodes[i][0]['normalized_log_prob']
            
            try:
                t_rank = ranked_endnodes_t_hat_str.index(t_str)
                norm_log_prob_delta = processed_endnodes[i][t_rank]['normalized_log_prob'] - beam_top_log_prob
            except ValueError: 
                norm_log_prob_delta = (processed_endnodes[i][-1]['normalized_log_prob'] - beam_top_log_prob) * self.off_beam_penalty
                
            norm_log_prob_deltas.append(- norm_log_prob_delta)

        mean_norm_log_prob_delta = - statistics.harmonic_mean(norm_log_prob_deltas)

        return mean_norm_log_prob_delta

class Rescorer():
    def __init__(self) -> None:
        pass
    
    def __call__(self, normalized_log_prob, reranker_score):
        raise NotImplementedError

class BatchedRescorer(Rescorer):
    def __init__(self) -> None:
        pass
    
    def __call__(self, 
        reranker_scores, # (N, Nc, 1)
        collated_normalized_log_prob # (N, Nc, 1)
    ):
        raise NotImplementedError

class BatchedLinearRescorer(BatchedRescorer):
    def __init__(self, original_log_prob_weight: float, reranker_weight: float):
        super().__init__()
        self.original_log_prob_weight = original_log_prob_weight
        self.reranker_weight = reranker_weight
        
    def __call__(self, 
        collated_normalized_log_prob, # (N, Nc, 1)
        reranker_scores, # (N, Nc, 1)
    ):
        return (self.original_log_prob_weight * collated_normalized_log_prob) + (self.reranker_weight * reranker_scores)

class LinearRescorer(Rescorer):
    def __init__(self, original_log_prob_weight: float, reranker_weight: float):
        super().__init__()
        self.original_log_prob_weight = original_log_prob_weight
        self.reranker_weight = reranker_weight
        
    def __call__(self, normalized_log_prob, reranker_score):
        return (self.original_log_prob_weight * normalized_log_prob) + (self.reranker_weight * reranker_score)

# adjusted score = (original_log_prob_weight * normalized_log_prob) + (reranker_weight * (reranker_score ** power))
class PolynomialRescorer(Rescorer):
    def __init__(self, original_log_prob_weight: float, reranker_weight: float, power: float):
        super().__init__()
        self.original_log_prob_weight = original_log_prob_weight
        self.reranker_weight = reranker_weight
        self.power = power
        
    def __call__(self, normalized_log_prob, reranker_score):
        return (self.original_log_prob_weight * normalized_log_prob) + (self.reranker_weight * (reranker_score ** self.power))
