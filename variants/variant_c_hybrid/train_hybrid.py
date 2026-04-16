"""
Variant C — Hybrid model with item metadata (LightFM).

Member: Sarat
Team 11, CMPE 256 - Recommender Systems

Approach
--------
LightFM learns user and item embeddings as the sum of embeddings of their
features. For items, we attach the `category` field from app_info_sample.csv
as a side feature. Items in the same category share information, which helps
the model generalize to less-popular / sparse items that pure CF struggles
with.

Leakage note
------------
app_info_sample.csv contains `installs`, `avg_rating`, and `rating_count`
that were collected DURING the interaction period (per the dataset authors'
README). Using them as features would leak future information into the
training signal, so we deliberately use ONLY `category`, which is fixed at
publish time.

Reference
---------
Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
Recommendations. Proceedings of the 2nd Workshop on New Trends on
Content-Based Recommender Systems (CBRecSys 2015).
arXiv:1507.08439
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# LightFM is imported lazily inside main() so the rest of the module
# (data loading, evaluation) is testable even if LightFM isn't installed.


# -------------------------------------------------------------------------
# Data loading — consumes the shared pipeline's outputs
# -------------------------------------------------------------------------

def load_splits(data_dir: Path):
    """Load train/val/test splits produced by the shared pipeline.

    Expected files (adjust these paths if your pipeline uses different names):
      - train.csv, val.csv, test.csv  with columns [user_id, item_id, timestamp]
      - user_id_map.csv, item_id_map.csv  (original -> encoded)
    """
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        missing = {"user_id", "item_id"} - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")

    return train, val, test


def load_item_metadata(metadata_path: Path, item_id_map_path: Path):
    """Load app_info_sample.csv and align with the encoded item IDs.

    We keep ONLY a single category column. The leaky columns (installs,
    rating, rating_count, etc.) are dropped on purpose — see module docstring.
    """
    meta = pd.read_csv(metadata_path)
    id_map = pd.read_csv(item_id_map_path)

    # App id column: public Myket sample uses `app_name`; other exports may use
    # `package_name` or `item_id`.
    for key in ("package_name", "app_name", "item_id"):
        if key in meta.columns:
            meta_key = key
            break
    else:
        raise ValueError(
            "Metadata CSV needs one of: package_name, app_name, item_id"
        )

    # English category when present (sample CSV uses category_en); else `category`.
    if "category_en" in meta.columns:
        meta = meta.rename(columns={"category_en": "category"})
    elif "category" not in meta.columns:
        raise ValueError("Metadata CSV needs category_en or category")

    # Pipeline outputs `item_id` (encoded); some teams use `encoded_item_id`.
    if "encoded_item_id" not in id_map.columns and "item_id" in id_map.columns:
        id_map = id_map.rename(columns={"item_id": "encoded_item_id"})
    elif "encoded_item_id" not in id_map.columns:
        raise ValueError("item map needs item_id or encoded_item_id")

    merged = meta.merge(
        id_map,
        left_on=meta_key,
        right_on="original_item_id",
        how="inner",
    )

    merged = merged[["encoded_item_id", "category"]].copy()

    # Fill missing categories with an explicit "unknown" bucket so every
    # item has at least one feature.
    merged["category"] = merged["category"].fillna("unknown").astype(str)

    return merged


# -------------------------------------------------------------------------
# Sparse matrix builders
# -------------------------------------------------------------------------

def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.coo_matrix:
    """Build a (n_users x n_items) implicit-feedback matrix.

    Every observed (user, item) pair is a 1.0. Duplicates are collapsed.
    """
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    rows = df["user_id"].to_numpy()
    cols = df["item_id"].to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items))


def build_item_features(meta_df: pd.DataFrame, n_items: int):
    """One-hot encode the `category` column into an (n_items x n_features) matrix.

    LightFM expects item features as a sparse matrix where row i corresponds
    to item i. To recover the pure-CF behaviour on items that also have
    features, we typically add an identity block so each item keeps its own
    learned embedding in addition to the shared category embedding. This is
    the standard LightFM-hybrid recipe.
    """
    # Category -> column index
    categories = sorted(meta_df["category"].unique())
    cat_to_col = {c: i for i, c in enumerate(categories)}

    n_cat = len(categories)
    # Block 1: identity (item i -> feature i). Ensures items without a category
    # in the metadata table still get a unique embedding.
    eye = sparse.identity(n_items, format="csr", dtype=np.float32)

    # Block 2: category indicators
    rows = meta_df["encoded_item_id"].to_numpy()
    cols = np.array([cat_to_col[c] for c in meta_df["category"]])
    data = np.ones(len(meta_df), dtype=np.float32)
    cat_block = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_cat)).tocsr()

    # Horizontal stack: [identity | category-onehot]
    features = sparse.hstack([eye, cat_block], format="csr")
    return features, categories


# -------------------------------------------------------------------------
# Evaluation — uses the shared metrics (Precision@K, Recall@K, NDCG@K)
# -------------------------------------------------------------------------

def precision_recall_ndcg_at_k(
    model,
    test_interactions: sparse.coo_matrix,
    train_interactions: sparse.coo_matrix,
    item_features: sparse.csr_matrix,
    k: int = 10,
):
    """Evaluate on the shared framework's three metrics.

    Implements the exact same protocol described in Day 6:
      - only evaluate users with >=1 test interaction
      - remove items the user has seen in training (no trivial recommendations)
      - top-K ranking over all remaining items
    """
    test_csr = test_interactions.tocsr()
    train_csr = train_interactions.tocsr()
    n_users, n_items = test_csr.shape

    precisions, recalls, ndcgs = [], [], []
    item_idx = np.arange(n_items)

    # Precompute ideal-DCG normalisers up to k
    log2 = np.log2(np.arange(2, k + 2))

    for u in range(n_users):
        true_items = set(test_csr[u].indices.tolist())
        if not true_items:
            continue

        scores = model.predict(u, item_idx, item_features=item_features)

        # Mask out items already seen in train
        seen = train_csr[u].indices
        if len(seen):
            scores[seen] = -np.inf

        # Top-K
        if k < n_items:
            top_k = np.argpartition(-scores, k)[:k]
            top_k = top_k[np.argsort(-scores[top_k])]
        else:
            top_k = np.argsort(-scores)

        hits = np.array([1 if i in true_items else 0 for i in top_k], dtype=np.float32)
        n_hit = hits.sum()

        precisions.append(n_hit / k)
        recalls.append(n_hit / len(true_items))

        dcg = (hits / log2).sum()
        ideal_hits = min(len(true_items), k)
        idcg = (1.0 / log2[:ideal_hits]).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with train.csv / val.csv / test.csv")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Path to app_info_sample.csv")
    parser.add_argument("--item-id-map", type=Path, required=True,
                        help="Path to item mapping CSV (e.g. item_mapping.csv)")
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-features", action="store_true",
                        help="Ablation: train LightFM WITHOUT category features")
    parser.add_argument("--components", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--loss", type=str, default="warp",
                        choices=["warp", "bpr", "logistic", "warp-kos"])
    parser.add_argument("--output", type=Path, default=Path("results_hybrid.json"))
    args = parser.parse_args()

    # Lazy import so the rest of the module is testable without LightFM.
    from lightfm import LightFM

    print("Loading splits...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    n_users = int(pd.concat([train_df, val_df, test_df])["user_id"].max()) + 1
    n_items = int(pd.concat([train_df, val_df, test_df])["item_id"].max()) + 1
    print(f"  users={n_users}  items={n_items}  train={len(train_df)}  "
          f"val={len(val_df)}  test={len(test_df)}")

    print("Building interaction matrices...")
    train_mat = build_interaction_matrix(train_df, n_users, n_items)
    val_mat = build_interaction_matrix(val_df, n_users, n_items)
    test_mat = build_interaction_matrix(test_df, n_users, n_items)

    if args.no_features:
        print("Ablation mode: training WITHOUT category features (pure LightFM MF).")
        item_features = None
    else:
        print("Loading item metadata (category only; leaky columns excluded)...")
        meta_df = load_item_metadata(args.metadata, args.item_id_map)
        item_features, categories = build_item_features(meta_df, n_items)
        print(f"  {len(categories)} categories, "
              f"item_features shape = {item_features.shape}")

    print(f"Training LightFM (loss={args.loss}, components={args.components}, "
          f"epochs={args.epochs})...")
    model = LightFM(loss=args.loss, no_components=args.components, random_state=42)
    model.fit(train_mat, item_features=item_features,
              epochs=args.epochs, num_threads=4, verbose=True)

    print("Evaluating on validation set...")
    val_results = precision_recall_ndcg_at_k(
        model, val_mat, train_mat, item_features, k=args.k
    )
    print(f"  VAL  P@{args.k}={val_results['precision@k']:.4f}  "
          f"R@{args.k}={val_results['recall@k']:.4f}  "
          f"NDCG@{args.k}={val_results['ndcg@k']:.4f}")

    print("Evaluating on test set...")
    # For test, exclude BOTH train and val from the candidate pool
    train_plus_val = (train_mat.tocsr() + val_mat.tocsr()).tocoo()
    test_results = precision_recall_ndcg_at_k(
        model, test_mat, train_plus_val, item_features, k=args.k
    )
    print(f"  TEST P@{args.k}={test_results['precision@k']:.4f}  "
          f"R@{args.k}={test_results['recall@k']:.4f}  "
          f"NDCG@{args.k}={test_results['ndcg@k']:.4f}")

    out = {
        "variant": "hybrid_lightfm",
        "member": "Sarat",
        "uses_metadata": not args.no_features,
        "hyperparams": {
            "loss": args.loss, "components": args.components,
            "epochs": args.epochs, "k": args.k,
        },
        "val": val_results,
        "test": test_results,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
