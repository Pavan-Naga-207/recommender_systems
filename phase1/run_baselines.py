from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

DATA_DIR = "phase1/processed"
EVAL_SPLIT = "test"  
TOP_K = 10
CF_NEIGHBORS = 50
SVD_COMPONENTS = 64
OUTPUT_CSV = os.path.join(DATA_DIR, "baseline_results.csv")

if EVAL_SPLIT not in {"val", "test"}:
    raise ValueError("EVAL_SPLIT must be either 'val' or 'test'.")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
eval_df = val_df if EVAL_SPLIT == "val" else test_df

num_users = int(max(train_df["user_id"].max(), eval_df["user_id"].max()) + 1)
num_items = int(max(train_df["item_id"].max(), eval_df["item_id"].max()) + 1)

train_rows = train_df["user_id"].to_numpy(dtype=np.int64)
train_cols = train_df["item_id"].to_numpy(dtype=np.int64)
train_data = np.ones(len(train_df), dtype=np.float32)
train_matrix = sparse.csr_matrix(
    (train_data, (train_rows, train_cols)),
    shape=(num_users, num_items),
    dtype=np.float32,
)
train_matrix.data[:] = 1.0

train_user_items = {
    int(u): set(items.astype(int))
    for u, items in train_df.groupby("user_id")["item_id"]
}
eval_user_items = {
    int(u): set(items.astype(int))
    for u, items in eval_df.groupby("user_id")["item_id"]
}

eligible_users = sorted(set(eval_user_items.keys()) & set(train_user_items.keys()))
if not eligible_users:
    raise ValueError("No users overlap between train split and evaluation split.")


def top_k_from_scores(scores, seen_items, k):
    scores = np.asarray(scores, dtype=np.float64).copy()
    if seen_items:
        seen_idx = np.fromiter(seen_items, dtype=np.int64)
        scores[seen_idx] = -np.inf

    finite_mask = np.isfinite(scores)
    valid_count = int(finite_mask.sum())
    if valid_count == 0:
        return []

    k_eff = min(k, valid_count)
    top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx.tolist()


def evaluate_topk(recommender, users, ground_truth, k, model_name):
    precisions = []
    recalls = []
    ndcgs = []

    for user_id in users:
        truth = ground_truth.get(user_id, set())
        if not truth:
            continue

        recs = recommender(user_id, k)
        if not recs:
            precisions.append(0.0)
            recalls.append(0.0)
            ndcgs.append(0.0)
            continue

        hits = [1 if item in truth else 0 for item in recs]
        hit_count = float(sum(hits))
        precision = hit_count / float(k)
        recall = hit_count / float(len(truth))

        dcg = 0.0
        for rank, rel in enumerate(hits, start=1):
            if rel:
                dcg += 1.0 / np.log2(rank + 1.0)

        ideal_hits = min(len(truth), k)
        idcg = sum(1.0 / np.log2(i + 1.0) for i in range(2, ideal_hits + 2))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    if not precisions:
        return {
            "model": model_name,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "users_evaluated": 0,
        }

    return {
        "model": model_name,
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "users_evaluated": len(precisions),
    }


# Baseline 1: Popularity
item_pop = np.asarray(train_matrix.sum(axis=0)).ravel()
popularity_ranking = np.argsort(item_pop)[::-1]


def recommend_popularity(user_id, k):
    seen = train_user_items.get(user_id, set())
    recs = []
    for item_id in popularity_ranking:
        if int(item_id) not in seen:
            recs.append(int(item_id))
        if len(recs) == k:
            break
    return recs


# Baseline 2: UserCF (cosine over binary interactions)
row_norms = np.sqrt(train_matrix.multiply(train_matrix).sum(axis=1)).A1
row_norms[row_norms == 0.0] = 1.0
train_matrix_norm = train_matrix.multiply(1.0 / row_norms[:, None]).tocsr()


def recommend_user_cf(user_id, k):
    seen = train_user_items.get(user_id, set())
    sim = train_matrix_norm[user_id].dot(train_matrix_norm.T).toarray().ravel()
    sim[user_id] = 0.0

    positive = np.where(sim > 0.0)[0]
    if positive.size == 0:
        return recommend_popularity(user_id, k)

    n_neighbors = min(CF_NEIGHBORS, positive.size)
    neigh_idx = positive[np.argpartition(sim[positive], -n_neighbors)[-n_neighbors:]]
    neigh_weights = sim[neigh_idx]
    neigh_matrix = train_matrix[neigh_idx].toarray()
    scores = neigh_weights @ neigh_matrix
    return top_k_from_scores(scores, seen, k)


# Baseline 3: Matrix factorization with functional sparse SVD.
max_components = min(train_matrix.shape) - 1
svd_components = max(2, min(SVD_COMPONENTS, max_components))
user_left, singular_vals, item_right_t = sparse_linalg.svds(train_matrix, k=svd_components)
order = np.argsort(singular_vals)[::-1]
singular_vals = singular_vals[order]
user_left = user_left[:, order]
item_right_t = item_right_t[order, :]
user_factors = user_left * singular_vals
item_factors = item_right_t.T


def recommend_svd(user_id, k):
    seen = train_user_items.get(user_id, set())
    scores = user_factors[user_id] @ item_factors.T
    return top_k_from_scores(scores, seen, k)


results = [
    evaluate_topk(recommend_popularity, eligible_users, eval_user_items, TOP_K, "Popularity"),
    evaluate_topk(recommend_user_cf, eligible_users, eval_user_items, TOP_K, "UserCF"),
    evaluate_topk(recommend_svd, eligible_users, eval_user_items, TOP_K, "SVD"),
]

result_df = pd.DataFrame(
    [
        {
            "model": r["model"],
            f"precision@{TOP_K}": r["precision_at_k"],
            f"recall@{TOP_K}": r["recall_at_k"],
            f"ndcg@{TOP_K}": r["ndcg_at_k"],
            "users_evaluated": r["users_evaluated"],
            "eval_split": EVAL_SPLIT,
        }
        for r in results
    ]
)

result_df.to_csv(OUTPUT_CSV, index=False)
print(result_df.to_string(index=False))
print(f"\nSaved baseline results to: {OUTPUT_CSV}")
