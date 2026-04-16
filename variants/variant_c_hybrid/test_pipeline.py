"""Smoke test for the non-LightFM parts of train_hybrid.py.

Validates that data loading, sparse matrix building, feature building,
and the evaluation loop all work correctly on a small synthetic dataset.
Run this before trying the real thing to catch plumbing bugs fast.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from train_hybrid import (
    build_interaction_matrix,
    build_item_features,
    load_item_metadata,
    load_splits,
    precision_recall_ndcg_at_k,
)

# -- 1. Build a tiny synthetic dataset --------------------------------------
rng = np.random.default_rng(0)
N_USERS, N_ITEMS = 50, 20
N_CAT = 4

# 200 random interactions
ints = pd.DataFrame({
    "user_id": rng.integers(0, N_USERS, 200),
    "item_id": rng.integers(0, N_ITEMS, 200),
    "timestamp": rng.integers(0, 1000, 200),
}).drop_duplicates(["user_id", "item_id"]).sort_values("timestamp")

# Split 80/10/10 temporally
n = len(ints)
train_df = ints.iloc[: int(0.8 * n)]
val_df = ints.iloc[int(0.8 * n) : int(0.9 * n)]
test_df = ints.iloc[int(0.9 * n):]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    train_df.to_csv(td / "train.csv", index=False)
    val_df.to_csv(td / "val.csv", index=False)
    test_df.to_csv(td / "test.csv", index=False)

    # -- 2. Test load_splits --------------------------------------------------
    t, v, te = load_splits(td)
    assert len(t) == len(train_df) and len(v) == len(val_df) and len(te) == len(test_df)
    print(f"[ok] load_splits  train={len(t)} val={len(v)} test={len(te)}")

    # -- 3. Test build_interaction_matrix ------------------------------------
    m = build_interaction_matrix(t, N_USERS, N_ITEMS)
    assert m.shape == (N_USERS, N_ITEMS)
    assert m.nnz <= len(t)  # duplicates collapsed
    print(f"[ok] build_interaction_matrix  shape={m.shape} nnz={m.nnz}")

    # -- 4. Test load_item_metadata (with a simulated leaky file) ------------
    leaky_meta = pd.DataFrame({
        "item_id": [f"app_{i}" for i in range(N_ITEMS)],
        "category": [f"cat_{i % N_CAT}" for i in range(N_ITEMS)],
        "installs": rng.integers(100, 1_000_000, N_ITEMS),  # LEAKY
        "avg_rating": rng.uniform(1, 5, N_ITEMS),            # LEAKY
        "rating_count": rng.integers(1, 10_000, N_ITEMS),    # LEAKY
    })
    id_map = pd.DataFrame({
        "original_item_id": [f"app_{i}" for i in range(N_ITEMS)],
        "item_id": list(range(N_ITEMS)),
    })
    meta_path = td / "app_info_sample.csv"
    map_path = td / "item_id_map.csv"
    leaky_meta.to_csv(meta_path, index=False)
    id_map.to_csv(map_path, index=False)

    meta = load_item_metadata(meta_path, map_path)
    assert "installs" not in meta.columns, "leaky column leaked through!"
    assert "avg_rating" not in meta.columns, "leaky column leaked through!"
    assert "rating_count" not in meta.columns, "leaky column leaked through!"
    assert set(meta.columns) == {"encoded_item_id", "category"}
    print(f"[ok] load_item_metadata  leaky columns excluded; kept={list(meta.columns)}")

# -- 5. Test build_item_features ---------------------------------------------
features, cats = build_item_features(meta, N_ITEMS)
# identity block + category block
assert features.shape == (N_ITEMS, N_ITEMS + len(cats))
assert len(cats) == N_CAT
# Each item should have exactly 2 non-zero features: its own identity + 1 category
rows_nnz = np.asarray(features.sum(axis=1)).flatten()
assert (rows_nnz == 2).all(), f"expected 2 features per item, got {rows_nnz}"
print(f"[ok] build_item_features  shape={features.shape}  2 features/item")

# -- 6. Test evaluation with a dummy model -----------------------------------
class DummyModel:
    """Pretends to be LightFM — returns random scores."""
    def predict(self, user_id, item_ids, item_features=None):
        return rng.standard_normal(len(item_ids))

train_mat = build_interaction_matrix(train_df, N_USERS, N_ITEMS)
test_mat = build_interaction_matrix(test_df, N_USERS, N_ITEMS)
res = precision_recall_ndcg_at_k(DummyModel(), test_mat, train_mat, features, k=5)
assert 0 <= res["precision@k"] <= 1
assert 0 <= res["recall@k"] <= 1
assert 0 <= res["ndcg@k"] <= 1
print(f"[ok] precision_recall_ndcg_at_k  "
      f"P@5={res['precision@k']:.3f} R@5={res['recall@k']:.3f} "
      f"NDCG@5={res['ndcg@k']:.3f} (random, so low is expected)")

print("\nAll smoke tests passed. The plumbing is correct.")
print("On your machine: pip install lightfm, then run train_hybrid.py for real.")
