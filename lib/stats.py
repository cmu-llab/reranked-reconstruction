from enum import Enum
import wandb
api = wandb.Api()
wandb.login()
import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
from scipy.stats import pearsonr
from dotenv import load_dotenv
import os
load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

# calculates difference in means
def diff_in_means(data1, data2):
    return np.mean(data1) - np.mean(data2)

# dataset
class D(Enum):
    NRO_IPA = "Nromance_ipa"
    NRO_ORTO = "Nromance_orto"
    BAXTER = "chinese_baxter"
    WIKIHAN = "chinese_wikihan2022"
    WIKIHAN_A = "chinese_wikihan2022_augmented"
allD = [D.WIKIHAN, D.WIKIHAN_A, D.BAXTER, D.NRO_IPA, D.NRO_ORTO]

# metric
class M(Enum):
    TED = "TED"
    TER = "TER"
    FER = "FER"
    ACC = "ACC"
    BCF = "BCF"
allM = [M.ACC, M.TED, M.TER, M.FER, M.BCF]

# system
class S(Enum):
    GRU = "GRU"
    TRANS = "Transformer"
    JGRU = "JambuGRU"
    JTRANS = "JambuTransformer"
    GRUBS = "GRU-BS"
    GRUBS_GRU = "GRU-BS+GRU"
    GRUBS_TRANS = "GRU-BS+Transformer"
    GRUBS_JTRANS = "GRU-BS+JambuTransformer"
allS = [s for s in S]
reflexSystems = [S.GRU, S.TRANS, S.JGRU, S.JTRANS]
reconSystems = [S.GRU, S.TRANS, S.GRUBS, S.GRUBS_GRU, S.GRUBS_TRANS, S.GRUBS_JTRANS]

# randomly selected runs used for error analysis
error_analysis_samples = {
    D.WIKIHAN: {
        S.GRUBS_GRU: '3rxb5sl0',
        S.GRUBS_TRANS: 'sgsp6lm3',
        S.GRUBS_JTRANS: 'd1mf4isq',
    },
    D.WIKIHAN_A: {
        S.GRUBS_GRU: 'iq79hdlv',
        S.GRUBS_TRANS: 'g4385tbt',
        S.GRUBS_JTRANS: 'cko3pyzr',
    },
    D.BAXTER: {
        S.GRUBS_GRU: 'miz4eine',
        S.GRUBS_TRANS: 'mwv3d0wo',
        S.GRUBS_JTRANS: 'g4tchsdo',
    },
    D.NRO_IPA: {
        S.GRUBS_GRU: 'ot4jmyyo',
        S.GRUBS_TRANS: 'xtymzpc0',
        S.GRUBS_JTRANS: 'ckw5kexw',
    },
    D.NRO_ORTO: {
        S.GRUBS_GRU: 'uoexslwz',
        S.GRUBS_TRANS: 'bwzh7cnr',
        S.GRUBS_JTRANS: 'lu2ll15j',
    },
}

from itertools import product
def get_pairs(L):
    all_ref_pairs = list(product(L, L))
    all_ref_pair_unique = list(filter(lambda x: x[0].value != x[1].value, all_ref_pairs))
    return all_ref_pair_unique

# task
class T(Enum):
    REC = "d2p" # reconstruction
    REF = "p2d" # reflex prediction

experiment_t = tuple[T, D, S]

baseline_res: dict[(T, D, S), dict[M, float]] = {
    (T.REC, D.BAXTER, S.TRANS): {
        M.TED: [0.96273292, 0.96273292, 0.97515528, 1.01242236, 0.9068323, 0.98757764, 0.9689441, 0.94409938, 1.03726708, 1.00621118, 1.08695652, 1.01242236, 0.95031056, 0.98757764, 1.04347826, 0.92546584, 0.98757764, 0.96273292, 1.03726708, 1.04968944],
        M.TER: [0.21283644, 0.21511387, 0.22432712, 0.22877847, 0.20776398, 0.22028986, 0.21977226, 0.2115942, 0.23561077, 0.2252588, 0.24813665, 0.22691511, 0.213147, 0.22111801, 0.23146998, 0.20838509, 0.22360248, 0.21511387, 0.23571429, 0.24089027],
        M.FER: [0.08179982, 0.081538, 0.08490425, 0.09414273, 0.07854578, 0.09694794, 0.08946738, 0.07873279, 0.08883154, 0.09029025, 0.1073459, 0.08658737, 0.08438061, 0.08460503, 0.09058947, 0.08026631, 0.08486685, 0.081538, 0.09414273, 0.09032765],
        M.ACC: [0.42857143, 0.39751553, 0.37888199, 0.36024845, 0.43478261, 0.37267081, 0.40993789, 0.42236025, 0.34782609, 0.37267081, 0.33540373, 0.39751553, 0.41614907, 0.40372671, 0.38509317, 0.42857143, 0.36024845, 0.39751553, 0.40372671, 0.34782609],
        M.BCF: [0.70121801, 0.70221467, 0.70025448, 0.69064947, 0.7161499, 0.69793146, 0.70733803, 0.70750469, 0.6830944, 0.69234814, 0.68355747, 0.69051482, 0.69854121, 0.69845064, 0.68151142, 0.70438881, 0.69666584, 0.70221467, 0.67880164, 0.6768576],
    },
    (T.REC, D.BAXTER, S.GRU): {
        M.TED: [1.1552795, 1.16770186, 1.07453416, 0.99378882, 1.08074534, 1.08695652, 1.11801242, 1.0621118, 1.16770186, 1.01242236, 1.0621118, 1.19254658, 1.16149068, 0.98136646, 1.06832298, 1.0310559, 1.02484472, 1.08695652, 1.1242236, 1.18012422],
        M.TER: [0.26149068, 0.2615942, 0.24668737, 0.22484472, 0.24606625, 0.24534161, 0.25113872, 0.24213251, 0.26428571, 0.22743271, 0.23964803, 0.27318841, 0.26656315, 0.22153209, 0.24409938, 0.23509317, 0.23250518, 0.24937888, 0.25724638, 0.26770186],
        M.FER: [0.09144973, 0.1005386, 0.09361909, 0.08490425, 0.09014063, 0.09014063, 0.09657391, 0.09429234, 0.09365649, 0.08722322, 0.08583932, 0.09328247, 0.09522741, 0.08411879, 0.0964991, 0.08486685, 0.08497905, 0.08849491, 0.09245961, 0.10001496],
        M.ACC: [0.34782609, 0.34161491, 0.31055901, 0.38509317, 0.36645963, 0.37888199, 0.34161491, 0.34161491, 0.33540373, 0.35403727, 0.34161491, 0.32919255, 0.27950311, 0.36645963, 0.36024845, 0.34161491, 0.36645963, 0.36645963, 0.32298137, 0.34782609],
        M.BCF: [0.64594368, 0.65607745, 0.6750964, 0.69145982, 0.66836846, 0.67388816, 0.66889929, 0.67645322, 0.65022867, 0.68527989, 0.67401579, 0.64656547, 0.65645441, 0.70410627, 0.67733082, 0.68503417, 0.69210135, 0.66604384, 0.6580116, 0.64180756],
    },
    (T.REC, D.NRO_IPA, S.TRANS): {
        M.TED: [0.92588369, 0.89623717, 0.92531357, 0.91733181, 0.88255416, 0.87172178, 0.88939567, 0.91277081, 0.9304447, 0.88825542, 0.92417332, 0.91163056, 0.91619156, 0.88426454, 0.91448119, 0.92132269, 0.89851767, 0.89623717, 0.89281642, 0.90079818],
        M.TER: [0.11758203, 0.1142083, 0.11510594, 0.11670063, 0.11309177, 0.11153246, 0.11264503, 0.11611671, 0.11803489, 0.11289661, 0.11709279, 0.11520711, 0.11601031, 0.11270179, 0.11639737, 0.11506592, 0.11340019, 0.1142083, 0.11338583, 0.11457333],
        M.FER: [0.03958479, 0.0370603, 0.03808467, 0.03805154, 0.03707488, 0.03629964, 0.03779048, 0.03849151, 0.03865186, 0.03616182, 0.03827815, 0.03819732, 0.03731606, 0.0366866, 0.03819334, 0.03856042, 0.03786469, 0.0370603, 0.03683634, 0.03760496],
        M.ACC: [0.52337514, 0.53306727, 0.52280502, 0.51824401, 0.54275941, 0.53762828, 0.53933865, 0.53534778, 0.52451539, 0.5290764, 0.52508552, 0.5313569, 0.52337514, 0.54047891, 0.51653364, 0.52565564, 0.52622577, 0.53306727, 0.54618016, 0.5336374],
        M.BCF: [0.83837129, 0.84228126, 0.83972068, 0.84051514, 0.84458413, 0.84681252, 0.84382214, 0.84025832, 0.83848113, 0.84473944, 0.83888452, 0.8405036, 0.83976902, 0.84515266, 0.8398054, 0.83943615, 0.84281885, 0.84228126, 0.84296178, 0.84276465],
    },
    (T.REC, D.NRO_IPA, S.GRU): {
        M.TED: [1.00171038, 1.01254276, 0.9663626, 1.02223489, 0.9663626, 0.94925884, 0.93785633, 0.96522235, 0.96750285, 0.98346636, 0.96978335, 0.96978335, 0.98346636, 0.99030787, 0.98802737, 0.99030787, 0.9977195, 0.98745724, 0.95153934, 0.94925884], 
        M.TER: [0.12822367, 0.12967882, 0.12287184, 0.13005026, 0.12287184, 0.12088919, 0.11972517, 0.12376912, 0.12351091, 0.12529164, 0.12391789, 0.12220989, 0.12448236, 0.12540776, 0.12499572, 0.12585107, 0.12643847, 0.12562121, 0.11986592, 0.12145587], 
        M.FER: [0.04078011, 0.04093649, 0.03817214, 0.04064892, 0.03817214, 0.03674358, 0.03679261, 0.03852464, 0.03933168, 0.03962322, 0.03937674, 0.03838682, 0.03953576, 0.03902026, 0.03974116, 0.03977032, 0.03910507, 0.03951058, 0.03779446, 0.03774278], 
        M.ACC: [0.51824401, 0.51653364, 0.52394527, 0.50342075, 0.52394527, 0.51482326, 0.53192702, 0.53021665, 0.51767389, 0.51995439, 0.52394527, 0.52337514, 0.51824401, 0.51596351, 0.51425314, 0.51311288, 0.51425314, 0.51425314, 0.52736602, 0.51824401], 
        M.BCF: [0.82381446, 0.82221345, 0.82899708, 0.82021549, 0.82899708, 0.83170844, 0.83396315, 0.82988654, 0.82909667, 0.82645185, 0.82890367, 0.82895706, 0.82663753, 0.82447947, 0.82664337, 0.82649983, 0.8245638, 0.82587557, 0.83166223, 0.83042597], 
    },
    (T.REC, D.NRO_ORTO, S.TRANS): {
        M.TED: [0.5626072, 0.59748428, 0.55174385, 0.56032018, 0.53802173, 0.56603774, 0.56946827, 0.55974843, 0.56032018, 0.55917667, 0.55060034, 0.54316752, 0.53916524, 0.57575758, 0.56032018, 0.55574614, 0.56317896, 0.59748428, 0.57232704, 0.58890795],
        M.TER: [0.07300361, 0.0781016, 0.07242725, 0.07331988, 0.07006397, 0.07425602, 0.07438932, 0.07188971, 0.07190379, 0.07309054, 0.07142653, 0.07027987, 0.07056315, 0.07427961, 0.07359194, 0.07346369, 0.07273367, 0.0781016, 0.07484134, 0.0765769],
        M.FER: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # not applicable
        M.ACC: [0.71297885, 0.70383076, 0.72212693, 0.71012007, 0.72212693, 0.71069182, 0.71012007, 0.71240709, 0.70954831, 0.71012007, 0.71640938, 0.70783305, 0.71126358, 0.70325901, 0.71069182, 0.7084048, 0.71069182, 0.70383076, 0.70783305, 0.70611778],
        M.BCF: [0.89855656, 0.8925952, 0.90026331, 0.89894635, 0.90255201, 0.89778312, 0.89659788, 0.89903971, 0.89850419, 0.89924944, 0.89996818, 0.90168966, 0.90182827, 0.89583362, 0.89840422, 0.89991456, 0.89809623, 0.8925952, 0.8966686, 0.89352963]
    },
    (T.REC, D.NRO_ORTO, S.GRU): {
        M.TED: [0.59748428, 0.60091481, 0.61292167, 0.57747284, 0.59748428, 0.59176672, 0.60091481, 0.59862779, 0.60891938, 0.60434534, 0.64551172, 0.58833619, 0.59634077, 0.6049171, 0.60091481, 0.6049171, 0.59862779, 0.59862779, 0.59119497, 0.58776444],
        M.TER: [0.07792381, 0.07847549, 0.08023009, 0.075017, 0.07732342, 0.07759639, 0.07797288, 0.07792705, 0.07976519, 0.0786362, 0.08403052, 0.07546577, 0.07609566, 0.07875791, 0.07802243, 0.07811551, 0.07781894, 0.07777944, 0.07838692, 0.07671026],
        M.FER: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # not applicable
        M.ACC: [0.68839337, 0.69068039, 0.68439108, 0.70097198, 0.69296741, 0.69411092, 0.69639794, 0.69582619, 0.69411092, 0.69639794, 0.67867353, 0.70040023, 0.69925672, 0.69639794, 0.69296741, 0.69353917, 0.69353917, 0.6969697, 0.69811321, 0.69868496],
        M.BCF: [0.89104704, 0.88994925, 0.88897113, 0.89458052, 0.89177385, 0.89246635, 0.88989974, 0.89120236, 0.88866861, 0.89046487, 0.88290206, 0.89280491, 0.89148584, 0.88931053, 0.88999101, 0.88985044, 0.89095331, 0.89057994, 0.89225506, 0.89298319],
    },
}

import wandb
api = wandb.Api()
wandb.login()
import numpy as np
from scipy.stats import ranksums
NUM_RUNS = 20

reranking_runs = api.runs(
    path = f'{WANDB_ENTITY}/{WANDB_PROJECT}',
    filters={
        "state": "finished",
        "tags": "reranking_eval2_fixed_beam_fixed_ratio",
    },
)

correlation_runs = api.runs(
    path = f'{WANDB_ENTITY}/{WANDB_PROJECT}',
    filters={
        "state": "finished",
        "tags": "reranking_eval2_worse_d2p_fixed_beam_fixed_ratio",
    },
)

pretrained_runs = api.runs(
    path = f'{WANDB_ENTITY}/{WANDB_PROJECT}',
    filters={
        "state": "finished",
        "tags": "from-config",
    },
)

ablation_runs = api.runs(
    path = f'{WANDB_ENTITY}/{WANDB_PROJECT}',
    filters={
        "state": "finished",
        "tags": "beam_size_adjustment",
    },
)

def mk_metrics_from_runs(task: T, runs):
    if (len(runs) != NUM_RUNS):
        print("WARNING: wrong number of runs for", task, "using", len(runs), "instead of", NUM_RUNS)
        assert False
    return {
        M.ACC: list(map(lambda run: run.summary[f'{task.value}/test/accuracy'], runs)),
        M.TED: list(map(lambda run: run.summary[f'{task.value}/test/phoneme_edit_distance'], runs)),
        M.TER: list(map(lambda run: run.summary[f'{task.value}/test/phoneme_error_rate'], runs)),
        M.FER: list(map(lambda run: run.summary[f'{task.value}/test/feature_error_rate'], runs)),
        M.BCF: list(map(lambda run: run.summary[f'{task.value}/test/bcubed_f_score'], runs)),
    }

def save_runs_to_cache(task: T, dataset: D, system: S, runs):
    filename = f"res_cache/{task.value}_{dataset.value}_{system.value}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(runs, f)

import os
import pickle

def fetch_cached_runs(task: T, dataset: D, system: S):
    filename = f"res_cache/{task.value}_{dataset.value}_{system.value}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def get_correlation_runs(dataset: D, system: S):
    task = T.REC
    
    match (task, system):
        case (T.REC, S.GRUBS_GRU | S.GRUBS_TRANS | S.GRUBS_JTRANS): # reranking
            match system:
                case S.GRUBS_GRU:
                    p2d_architecture = 'GRU'
                case S.GRUBS_TRANS:
                    p2d_architecture = 'Transformer'
                case S.GRUBS_JTRANS:
                    p2d_architecture = 'JambuTransformer'
            def p(run):
                return True \
                    and run.config['dataset'] == dataset.value \
                    and run.config['p2d_architecture'] ==  p2d_architecture \
                    and not 'dev' in run.tags \

            dep_runs = list(filter(p, correlation_runs))

        case _:
            raise Exception("No correlation runs for this system")
           
    indep_runs = list(map(lambda run: api.run(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{run.config["p2d_run"]}'), dep_runs))

    indep_runs_res = mk_metrics_from_runs(T.REF, indep_runs)
    dep_runs_res = mk_metrics_from_runs(task, dep_runs)
    
    return indep_runs_res, dep_runs_res

def run_correlation_test(datasets: list[D], systems: list[S]) -> None:
    results: dict[D, S, dict[M, dict]] = {}
    for d in datasets:
        for s in systems:
            results[(d, s)] = {}
            indep_runs, dep_runs = get_correlation_runs(d, s)
            for m in allM:
                pearsonr_res = pearsonr(indep_runs[m], dep_runs[m])
                corr_coefficient = pearsonr_res.statistic
                pval = pearsonr_res.pvalue
                results[(d, s)][m] = {
                    'corr_coefficient': corr_coefficient, 
                    'invalidate': d == D.NRO_ORTO and m == M.FER,
                    'pval': pval, 
                    'metric': m,
                    'dataset': d, 
                }
    return results

# get runs from WandB
def get_runs(task: T, dataset: D, system: S) -> dict[M, list[float]]:
    if (task, dataset, system) in baseline_res:
        return baseline_res[(task, dataset, system)]
    
    # look up cache
    runs = fetch_cached_runs(task, dataset, system)
    if runs is not None:
        return runs
    
    # else fetch
    match (task, system):
        case (T.REF, _):
            def p(run):
                return True \
                    and run.config['dataset'] == dataset.value \
                    and run.config['architecture'] == system.value \
                    and run.config['submodel'] == task.value \
                    and not 'dev' in run.tags \

            runs = list(filter(p, pretrained_runs))
            
        case (T.REC, S.GRU): # wikihan GRU recon
            assert dataset == D.WIKIHAN or dataset == D.WIKIHAN_A
            def p(run):
                return True \
                    and run.config['dataset'] == dataset.value \
                    and run.config['architecture'] == system.value \
                    and run.config['submodel'] == task.value \
                    and not 'dev' in run.tags \
                    and run.config['d2p_decode_mode'] == 'greedy_search'
            
            runs = list(filter(p, pretrained_runs))
        
        case (T.REC, S.TRANS): # wikihan Trans recon
            assert dataset == D.WIKIHAN or dataset == D.WIKIHAN_A
            def p(run):
                return True \
                    and run.config['dataset'] == dataset.value \
                    and run.config['architecture'] == system.value \
                    and run.config['submodel'] == task.value \
                    and not 'dev' in run.tags 
            
            runs = list(filter(p, pretrained_runs))
           
        case (T.REC, S.GRUBS): # ablation
            def p(run):
                return True \
                    and run.config['dataset'] == dataset.value \
                    and not 'dev' in run.tags \
                    and run.config['new_beam_size'] == 10
  
            runs = list(filter(p, ablation_runs))

        case (T.REC, S.GRUBS_GRU | S.GRUBS_TRANS | S.GRUBS_JTRANS): # reranking
            match system:
                case S.GRUBS_GRU:
                    p2d_architecture = 'GRU'
                case S.GRUBS_TRANS:
                    p2d_architecture = 'Transformer'
                case S.GRUBS_JTRANS:
                    p2d_architecture = 'JambuTransformer'
            def p(run):
                return True \
                    and run.config['dataset'] == dataset.value \
                    and run.config['p2d_architecture'] ==  p2d_architecture \
                    and not 'dev' in run.tags \

            runs = list(filter(p, reranking_runs))

        case _:
            raise Exception("No results for this system")

    runs = mk_metrics_from_runs(task, runs)

    # save cache
    save_runs_to_cache(task, dataset, system, runs)

    return runs


def run_tests(task: T, dataset: D, pairs: list[tuple[S, S]]):
    results = {}
    for (s1, s2) in pairs:
        runs1 = get_runs(task = task, dataset = dataset, system = s1)
        runs2 = get_runs(task = task, dataset = dataset, system = s2)
        for m in allM:
            results[(s1, s2)] = {}
        for m in allM:
            match m:
                case M.ACC | M.BCF:
                    tail = "greater"
                case M.TER | M.TED | M.FER:
                    tail = "less"
            ranksum_res = ranksums(runs1[m], runs2[m], alternative = tail)
            bootstrap_res = stats.bootstrap((runs1[m], runs2[m]), diff_in_means, confidence_level=0.99, random_state=0)
            
            sig_better, pval_sig, ci_sig = significantly_better(ranksum_res, bootstrap_res, m)

            if sig_better:
                assert ranksum_res.pvalue < 0.01
                assert bootstrap_res.confidence_interval.low * bootstrap_res.confidence_interval.high > 0
            
            results[(s1, s2)][m] = {
                'ranksum': ranksum_res, 
                'bootstrap': bootstrap_res, 
                'sig_better': sig_better, 
                'invalidate': dataset == D.NRO_ORTO and m == M.FER,
                'pval_sig': pval_sig,
                'ci_sig': ci_sig,
                'metric': m,
                'dataset': dataset, 
            }
            
    return results

# the res should come from testing model 1 against 2, and return whether model 1 is better than model 2 on the specified metric
def significantly_better(ranksum_res, bootstrap_res, metric: M) -> bool:
    
    pval = ranksum_res.pvalue
    lo = bootstrap_res.confidence_interval.low
    hi = bootstrap_res.confidence_interval.high
    
    pval_sig = (pval < 0.01)

    match metric:
        case M.ACC | M.BCF:
            ci_sig = (lo > 0.0 and hi > 0.0)
        case M.TER | M.TED | M.FER:
            ci_sig = (lo < 0.0 and hi < 0.0)
            
    return pval_sig and ci_sig, pval_sig, ci_sig

def summary_stats(task: T, datasets: list[D], systems: list[S]) -> None:
    results = {}
    for d in datasets:
        for s in systems:
            runs = get_runs(task = task, dataset = d, system = s)
            for m in allM:
                results[(d, s)] = {}
            for m in allM:
                mean = np.mean(runs[m])
                std = np.std(runs[m])
                nsamples = len(runs[m])
                results[(d, s)][m] = {
                    'mean': mean, 
                    'std': std, 
                    'nsamples': nsamples, 
                    'invalidate': d == D.NRO_ORTO and m == M.FER,
                    'metric': m,
                    'dataset': d, 
                }
    return results