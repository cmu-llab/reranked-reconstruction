# Reranked Neural Protoform Reconstruction

This repository accompanies the paper Improved Neural Protoform Reconstruction via Reflex Prediction at LREC-COLING 2024. An accompanying appendix is also available as `appendix.pdf`.

![Diagram showing reranking in action via reflex prediction on protoform reconstruction candidates](https://share.cleanshot.com/7Mwvr1GQYFjSmQtdLft2+)

> **Abstract:** Protolanguage reconstruction is a central task in historical linguistics. The comparative method, one of the most influential theoretical and methodological frameworks in the history of the language sciences, allows linguists to infer protoforms (reconstructed ancestral words) from their reflexes (related modern words) based on the assumption of regular sound change. Not surprisingly, numerous computational linguists have attempted to operationalize comparative reconstruction through computational models. While these models have taken various forms, the most successful have been supervised encoder-decoder models, which treat the problem of predicting protoforms given sets of reflexes as a sequence-to-sequence problem. We argue that this framework ignores one of the most important aspects of the comparative method: not only should protoforms be inferable from cognate sets (sets of related reflexes) but the reflexes should be inferable from the protoforms. Leveraging another line of research—reflex prediction—we propose a system in which candidate protoforms from a reconstruction model are reranked by a reflex prediction model. We show that this more complete implementation of the comparative method allows us to surpass state-of-the-art protoform reconstruction methods on three of four Chinese and Romance datasets.

🚧 This repository is under construction. We plan on making more checkpoints available.

# Set up

## Python package requirements

```txt
editdistance==0.6.2
einops==0.6.1
huggingface-hub==0.16.4
lightning-utilities==0.8.0
lingpy==2.6.9
lingrex==1.3.0
matplotlib==3.7.1
numpy==1.24.3
pandas==2.0.2
panphon @ git+https://github.com/dmort27/panphon.git@6acd3833743a49e63941a0b740ee69eae1dafc1c
Pillow==9.4.0
pytorch-lightning==2.0.4
sacrebleu==2.3.1
seaborn==0.12.2
tabulate==0.9.0
tokenizers==0.13.3
toml==0.10.2
torch==2.0.1
torchaudio==2.0.2
torchmetrics==0.11.4
torchshow==0.5.0
torchvision==0.15.2
tqdm==4.65.0
transformers==4.31.0
wandb==0.15.3
python-dotenv==1.0.0
```

## GPU support

Cuda GPU is recommended. Running the experiments with a `--cpu` flag will attempt to run on them the CPU.

## WandB

All experiments need rely on [WandB](https://wandb.ai/) for results logging and checkpointing[^1]. To set up WandB, create a `.env` file with your WandB entity and project in the following format:

```txt
WANDB_ENTITY = "awandbentity"
WANDB_PROJECT = "awandbproject"
```

[^1]: We used the following tagging taxonomy to identify experiments on WandB:
    - `from-config` pretrains a reconstruction or reflex-prediction model
    - `beam_size_adjustment` evaluates a GRU-BS model under a different beam size
    - `reranking_eval2_fixed_beam_fixed_ratio` runs reranking experiment with a fixed beam size and score adjustment constant
    - `reranking_eval2_worse_d2p_fixed_beam_fixed_ratio` runs reranking experiment with a fixed beam size and score adjustment constant with the lowest performing reconstruction model

## Datasets

We used the following dataset naming convention in our code:

- WikiHan - `chinese_wikihan2022` or `wikihan`
- WikiHan-aug - `chinese_wikihan2022_augmented` or `wikihan_augmented`
- Hóu - `baxter`
- Rom-phon - `Nromance_ipa`[^2]
- Rom-orth - `Nromance_orto`[^2]

[^2]: The prefix `N` has no meaning other than identifying it as the same version used by Kim et al. (2023)

The Romance datasets is not available for redistribution. Please contact Ciobanu and Dinu (2014) to obtain their data.

# Evaluating Checkpoints

`run_checkpoints.ipynb` provides a walkthrough on loading our checkpoints and evaluating a reranked reconstruction system as well as its components.

# Training

In code, reflex prediction is called `p2d`, and reconstruction is called `d2p`.

## Baseline Reconstruction Models

Please refer to Kim et al. (2023). We transfer the results obtained with their into `lib/stats.py`.

## Training GRU-BS

```sh
python main.py --config best_hparams/d2p-wikihan-GRU_beam.pkl # WikiHan
python main.py --config best_hparams/d2p-wikihan_augmented-GRU_beam.pkl # WikiHan-aug
python main.py --config best_hparams/d2p-baxter-GRU_beam.pkl # Hóu
python main.py --config best_hparams/d2p-Nromance_ipa-GRU_beam.pkl # Rom-phon
python main.py --config best_hparams/d2p-Nromance_orto-GRU_beam.pkl # Rom-orth
```

## Evaluating GRU-BS at a different beam size

```sh
python test_with_different_beam_size.py <ID of GRU-BS training run>
# For example,
python test_with_different_beam_size.py dat1za2h
```

## Training reflex prediction models

```sh
# WikiHan
python main.py --config best_hparams/p2d-wikihan-GRU.pkl # baseline GRU
python main.py --config best_hparams/p2d-wikihan-JambuGRU.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-wikihan-JambuTransformer.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-wikihan-Transformer.pkl # Kim et al. (2023)'s Transformer

# WikiHan-aug
python main.py --config best_hparams/p2d-wikihan_augmented-GRU.pkl # baseline GRU
python main.py --config best_hparams/p2d-wikihan_augmented-JambuGRU.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-wikihan_augmented-JambuTransformer.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-wikihan_augmented-Transformer.pkl # Kim et al. (2023)'s Transformer

# Hóu
python main.py --config best_hparams/p2d-baxter-GRU.pkl # baseline GRU
python main.py --config best_hparams/p2d-baxter-JambuGRU.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-baxter-JambuTransformer.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-baxter-Transformer.pkl # Kim et al. (2023)'s Transformer

# Rom-phon
python main.py --config best_hparams/p2d-Nromance_ipa-GRU.pkl # baseline GRU
python main.py --config best_hparams/p2d-Nromance_ipa-JambuGRU.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-Nromance_ipa-JambuTransformer.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-Nromance_ipa-Transformer.pkl # Kim et al. (2023)'s Transformer

# Rom-orth
python main.py --config best_hparams/p2d-Nromance_orto-GRU.pkl # baseline GRU
python main.py --config best_hparams/p2d-Nromance_orto-JambuGRU.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-Nromance_orto-JambuTransformer.pkl # Arora et al. (2023)'s GRU
python main.py --config best_hparams/p2d-Nromance_orto-Transformer.pkl # Kim et al. (2023)'s Transformer
```

# Reranked Reconstruction

## Running reranked reconstruction grid search

```sh
python reranking_grid_search.py <reflex prediction model run ID> <reconstruction model run ID>
# for example:
python reranking_grid_search.py erifn8su t0h5ixg4
```

## Running reranked reconstruction evaluation

```sh
python reranking_eval.py <reflex prediction model run ID> <reconstruction model run ID>
# for example:
python reranking_eval.py r2yuitnv s8re3etu
```

## Running reranked reconstruction correlation experiment

```sh
python reranking_correlation.py <worse-performing reflex prediction model run ID> <reconstruction model run ID>
# for example:
python reranking_correlation.py t93lt8re zy9texe3
```

# Statistics

See `stats.ipynb`. Cached results used for statistics are under `res_cache`.
