# Phase 1: Shared Pipeline and Baselines

This folder contains the shared foundation for the project. Everything in `phase2/` builds on the processed files and evaluation setup produced here.

## Files

- `preprocess.py`  
  Loads the raw Myket interaction file, keeps the interaction triplet (`user_id`, `item_id`, `timestamp`), cleans the data, filters low-count users/items, encodes ids, and writes the train/validation/test splits.

- `common.py`  
  Shared utilities for loading processed splits, building the train interaction matrix, masking train-seen items, computing top-K metrics, and saving result CSVs.

- `run_baselines.py`  
  Runs the shared baselines: Popularity, UserCF, and SVD. These are the reference points for the phase 2 variants.

## How to run

From the repo root:

```bash
python phase1/preprocess.py
python phase1/run_baselines.py
```

The preprocessing script writes outputs to:

```text
phase1/processed/
```

The baseline script writes result CSVs to:

```text
phase2/results/
```

## What the pipeline does

The preprocessing step:

1. reads `myket-android-application-market-dataset/myket.csv`
2. keeps only `user_id`, `item_id`, and `timestamp`
3. removes duplicate and invalid rows
4. filters users and items with fewer than 5 interactions
5. encodes users/items into integer ids
6. sorts interactions by timestamp
7. creates a global temporal 80/10/10 train/val/test split

The final processed files include:

- `train.csv`
- `val.csv`
- `test.csv`
- `interaction_matrix.npy`
- `user_mapping.csv`
- `item_mapping.csv`
- `dataset_stats.json`
- `schema.json`

## Evaluation contract

All models use the same top-K evaluation setup:

- only users with both train history and validation/test interactions are evaluated
- items already seen in train are masked before recommendations are selected
- metrics are `Precision@10`, `Recall@10`, and `NDCG@10`

This keeps the baselines and all phase 2 variants comparable.

