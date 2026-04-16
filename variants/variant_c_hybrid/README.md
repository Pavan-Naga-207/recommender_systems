# Variant C — Hybrid Model with Item Metadata (LightFM)

**Team 11, CMPE 256 — Member: Sarat**

## Approach

This variant uses [LightFM](https://making.lyst.com/lightfm/docs/home.html)
(Kula, 2015 — [arXiv:1507.08439](https://arxiv.org/abs/1507.08439)) to build a
hybrid recommender that combines collaborative filtering with item-side
metadata.

LightFM represents each item as the **sum of embeddings of its features**. We
attach the English category (`category_en` in the public sample CSV) from
`app_info_sample.csv` to each item, so apps
in the same category share information. This helps the model generalize to
less-popular / sparse items that pure collaborative filtering struggles with.

## Why only `category`?

The sample `app_info_sample.csv` also contains `installs`, `rating`, and
`rating_count`. Per the dataset authors, these were collected **during the
interaction period**, which means using them as features would leak future
information into the training signal.

We deliberately use **only** the category column (`category_en`, aliased
internally to `category`), which is fixed at publish time and is not leaky.
The loader (`load_item_metadata`) keeps only category and encoded item id.

## How this differs from the two baselines

| Signal used       | Popularity | User-CF | This variant |
|-------------------|:----------:|:-------:|:------------:|
| Global item popularity | ✓     |         |              |
| User–user similarity   |       | ✓       |              |
| Learned latent factors |       |         | ✓            |
| Item category metadata |       |         | ✓            |

## How this fits the shared framework

- Consumes the pipeline's `train.csv`, `val.csv`, `test.csv` unchanged.
- Uses the same evaluation: **Precision@10, Recall@10, NDCG@10**.
- Uses the same leave-out protocol (items seen in train are removed from
  recommendation candidates; only users with ≥1 test interaction are scored).

## Files

- `train_hybrid.py` — main training + evaluation script
- `test_pipeline.py` — smoke tests for the non-LightFM parts (run this first)
- `requirements.txt` — dependencies

## Running

**Python version:** LightFM currently installs reliably on **Python 3.10–3.12**. On **3.13**, `pip install lightfm` often fails to build. Create a lower-Python env first, for example:

```bash
conda create -n hybrid_rec python=3.11 -y
conda activate hybrid_rec
```

```bash
# 1. Install deps (one-time)
pip install -r requirements.txt

# 2. Sanity-check the plumbing on synthetic data
python test_pipeline.py

# 3. Train + evaluate with category metadata
python train_hybrid.py \
    --data-dir path/to/shared_pipeline_outputs/ \
    --metadata path/to/app_info_sample.csv \
    --item-id-map path/to/item_mapping.csv

# 4. Ablation: same model but without category features (pure MF)
python train_hybrid.py \
    --data-dir path/to/shared_pipeline_outputs/ \
    --metadata path/to/app_info_sample.csv \
    --item-id-map path/to/item_mapping.csv \
    --no-features \
    --output results_no_features.json
```

The ablation run is what shows whether the metadata is actually helping.
Compare `results_hybrid.json` vs `results_no_features.json` on NDCG@10 — the
delta is the clean, attributable contribution of the `category` feature.

## Hyperparameters to try

| Flag              | Default | Try also          |
|-------------------|---------|-------------------|
| `--loss`          | warp    | bpr, logistic     |
| `--components`    | 64      | 16, 32, 128       |
| `--epochs`        | 20      | 10, 30, 50        |

WARP loss is a strong default for implicit-feedback top-K ranking — it
directly optimizes for ranking order, which is what our metrics reward.

## Reference

Kula, M. (2015). *Metadata Embeddings for User and Item Cold-start
Recommendations*. CBRecSys 2015. arXiv:1507.08439.
