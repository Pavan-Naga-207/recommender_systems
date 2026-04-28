# Phase 2: Variants and Comparisons

This folder contains the advanced recommendation variants built on top of the shared Phase 1 pipeline.

All scripts read processed data from:

```text
phase1/processed/
```

and write result CSVs to:

```text
phase2/results/
```

## Retained Interaction-Only Variants

- `run_ease.py`  
  Runs EASE, a regularized linear item-item recommender. This is the strongest plain interaction-only model in the repo.

- `run_rp3beta.py`  
  Runs RP3beta, a graph-based recommender over the user-item interaction graph. It uses `alpha` to shape random-walk transitions and `beta` to penalize very popular items.

- `run_itemknn.py`  
  Runs ItemKNN with BM25 weighting and `K=320` neighbors. This is the retained ItemKNN variant.

Run them from the repo root:

```bash
python phase2/run_ease.py
python phase2/run_rp3beta.py
python phase2/run_itemknn.py
python phase2/compare_results.py
```

## Metadata-Aware Variants

- `metadata_utils.py`  
  Shared helpers for loading item metadata, building metadata feature matrices, creating metadata-based item similarity graphs, and evaluating hybrid configs.

- `run_ease_metadata.py`  
  Runs a metadata sweep for EASE using metadata similarity and either score blending or reranking.

- `run_ease_metadata_best.py`  
  Runs the best pushed EASE metadata path: base EASE plus a small rating-prior term.

- `run_rp3beta_metadata.py`  
  Runs a metadata sweep for RP3beta using category / package-name metadata similarity and score blending or reranking.

- `run_rp3beta_metadata_best.py`  
  Runs the strongest pushed RP3beta metadata path. It augments the graph with dense item features, adds a category-content score, and adds a small item prior.

Run them from the repo root:

```bash
python phase2/run_ease_metadata.py
python phase2/run_ease_metadata_best.py
python phase2/run_rp3beta_metadata.py
python phase2/run_rp3beta_metadata_best.py
```

## Result Files

Key result files include:

- `results/baseline_results_val.csv`
- `results/baseline_results_test.csv`
- `results/comparison_retained_val.csv`
- `results/comparison_retained_test.csv`
- `results/ease_metadata_best_test.csv`
- `results/rp3beta_metadata_best_test.csv`

The notebooks in the repo read these CSVs directly for the walkthrough and final comparison.

## Main Results

All numbers below are from the test split with the shared Phase 1 evaluation setup.

| Model | Type | Precision@10 | Recall@10 | NDCG@10 | Source |
|---|---|---:|---:|---:|---|
| `EASEMeta-ratingPrior-a0.03` | EASE + metadata | 0.044929 | 0.066133 | 0.086694 | `results/ease_metadata_best_test.csv` |
| `RP3betaMetaLowRank-categoryContent` | RP3beta + metadata | 0.043602 | 0.066083 | 0.086520 | `results/rp3beta_metadata_best_test.csv` |
| `EASE-Binary-3000` | EASE | 0.044746 | 0.065984 | 0.086413 | `results/comparison_retained_test.csv` |
| `RP3beta-a0.9-b0.4-t400` | RP3beta | 0.043121 | 0.064997 | 0.085032 | `results/comparison_retained_test.csv` |
| `RP3betaMeta-category_char-rerank-a0.1` | RP3beta + metadata sweep | 0.043404 | 0.064650 | 0.084975 | `results/rp3beta_metadata_test_best.csv` |
| `ItemKNN-BM25-K320` | ItemKNN | 0.042994 | 0.063505 | 0.083244 | `results/comparison_retained_test.csv` |
| `UserCF` | baseline | 0.042754 | 0.063452 | 0.081960 | `results/comparison_retained_test.csv` |
| `Popularity` | baseline | 0.037797 | 0.058550 | 0.071684 | `results/comparison_retained_test.csv` |
| `SVD` | baseline | 0.022797 | 0.027715 | 0.038456 | `results/comparison_retained_test.csv` |

The best overall test result is `EASEMeta-ratingPrior-a0.03`. The strongest RP3beta result is `RP3betaMetaLowRank-categoryContent`, which is very close to the best EASE result and improves over the plain RP3beta anchor.

## Current Model Story

The main comparison is:

- **EASE**: global linear item-item reconstruction.
- **RP3beta**: graph-based random walk with popularity control.
- **ItemKNN BM25**: local item-neighborhood voting after BM25 weighting.

The latest metadata work keeps the strong interaction models as anchors and adds side information carefully. The best current results come from `EASEMeta-ratingPrior-a0.03` and `RP3betaMetaLowRank-categoryContent`.

