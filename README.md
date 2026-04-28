# Myket App Recommendation Project

This repo is our CMPE 256 recommender systems project on the Myket Android app install dataset. The task is to recommend unseen Android apps to users from their past install history. Since the data is implicit feedback, an observed install is treated as a positive signal, and non-observed apps are treated as unknown candidates rather than explicit negatives.

The easiest way to understand the project is to go through the notebooks in order. They were written as a walkthrough of the full project: dataset, preprocessing, baselines, each variant, metadata additions, and the final comparison.

## Recommended Walkthrough

1. `01_dataset_and_task.ipynb`  
   Starts with the raw Myket dataset, explains the recommendation task, summarizes dataset statistics, and visualizes the popularity skew / long-tail behavior.

2. `02_shared_pipeline_and_eval.ipynb`  
   Explains the shared Phase 1 pipeline: cleaning, minimum-interaction filtering, encoded ids, temporal 80/10/10 split, eligible users, train-seen item masking, and the shared metrics.

3. `03_baselines.ipynb`  
   Covers the shared baselines: Popularity, UserCF, and SVD. These are the reference points for judging whether the individual variants actually improve anything.

4. `04_rp3beta_deep_dive.ipynb`  
   Walks through the RP3beta variant in detail. It starts from the plain graph-based RP3beta model, checks nearby hyperparameters, adds a simple category-blend metadata prototype, and then compares against the pushed metadata-aware RP3beta implementations.

5. `05_ease_itemknn_and_metadata.ipynb`  
   Covers the teammate variants and metadata paths: plain EASE, EASE with metadata, ItemKNN BM25, and the earlier LightFM metadata experiment that was later replaced by ItemKNN BM25 as the retained Variant C model.

6. `06_final_comparison_and_report_notes.ipynb`  
   Pulls the results together into final comparison tables and report-ready notes.

## Repository Structure

```text
myket-android-application-market-dataset/   Raw dataset and app metadata
phase1/                                     Shared preprocessing and baseline code
phase1/processed/                           Generated train/val/test splits and mappings
phase2/                                     Retained model runners and metadata variants
phase2/results/                             Generated result CSVs
variants/variant_c_hybrid/                  Earlier Variant C experiments
analysis/                                   Small helpers used by the notebooks
```

## Main Scripts

Shared foundation:

```bash
python phase1/preprocess.py
python phase1/run_baselines.py
```

Retained interaction-only variants:

```bash
python phase2/run_ease.py
python phase2/run_rp3beta.py
python phase2/run_itemknn.py
python phase2/compare_results.py
```

Metadata-aware variants:

```bash
python phase2/run_ease_metadata.py
python phase2/run_ease_metadata_best.py
python phase2/run_rp3beta_metadata.py
python phase2/run_rp3beta_metadata_best.py
```

Note: the generated CSV outputs used in the notebooks are already under `phase2/results/`. If rerunning the dense-feature metadata scripts, make sure the expected dataset feature artifacts are available under `myket-android-application-market-dataset/data_int_index/`.

## Final Model Story

We compare three distinct recommender families:

- **EASE**: a global linear item-item recommender. Plain EASE is already very strong, and the metadata-aware rating-prior version gives a small additional gain.
- **RP3beta**: a graph-based random-walk recommender with a popularity penalty. The notebook shows the progression from plain RP3beta to a simple category-blend prototype and then to the stronger pushed metadata-aware version.
- **ItemKNN BM25**: a local item-neighborhood recommender using BM25 weighting. Variant C initially explored LightFM with metadata, but ItemKNN BM25 was retained because it produced stronger ranking results.

All models are evaluated under the same shared protocol using `Precision@10`, `Recall@10`, and `NDCG@10`.

## Current Best Results

The strongest test results from the current repo are:

| Model | Type | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|
| EASEMeta-ratingPrior-a0.03 | EASE + metadata | 0.044929 | 0.066133 | 0.086694 |
| RP3betaMetaLowRank-categoryContent | RP3beta + metadata | 0.043602 | 0.066083 | 0.086520 |
| EASE-Binary-3000 | EASE | 0.044746 | 0.065984 | 0.086413 |
| RP3beta-a0.9-b0.4-t400 | RP3beta | 0.043121 | 0.064997 | 0.085032 |
| ItemKNN-BM25-K320 | ItemKNN | 0.042994 | 0.063505 | 0.083244 |

The main takeaway is that interaction structure carries most of the signal in this dataset, while metadata gives small but useful improvements when added carefully.

## Notes on Metadata

The dataset includes app metadata such as installs, rating, rating count, and category. The aggregate fields can introduce temporal leakage if used carelessly, because they were collected during the interaction period. Category metadata is the safest side information, and the notebooks explain how metadata was incorporated and compared across variants.

