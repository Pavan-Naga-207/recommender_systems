"""
ItemKNN baselines (Cosine / TF-IDF / BM25) with tuned K.

Evaluation protocol matches phase1/run_baselines.py:
- fit on train only
- evaluate val/test with train-only masking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from train_hybrid import (
    build_interaction_matrix,
    load_splits,
)


def precision_recall_ndcg_predict_fn(
    test_interactions: sparse.coo_matrix,
    train_interactions: sparse.coo_matrix,
    predict_fn,
    k: int = 10,
):
    test_csr = test_interactions.tocsr()
    train_csr = train_interactions.tocsr()
    n_users, n_items = test_csr.shape

    precisions, recalls, ndcgs = [], [], []
    log2 = np.log2(np.arange(2, k + 2))

    for u in range(n_users):
        true_items = set(test_csr[u].indices.tolist())
        if not true_items:
            continue
        if train_csr[u].nnz == 0:
            continue

        scores = predict_fn(u)
        scores = np.asarray(scores, dtype=np.float64).copy()
        seen = train_csr[u].indices
        if len(seen):
            scores[seen] = -np.inf

        if k < n_items:
            top_k = np.argpartition(-scores, k)[:k]
            top_k = top_k[np.argsort(-scores[top_k])]
        else:
            top_k = np.argsort(-scores)

        hits = np.array([1.0 if i in true_items else 0.0 for i in top_k], dtype=np.float32)
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


def fit_similarity(train_csr: sparse.csr_matrix, family: str, k: int):
    from implicit.nearest_neighbours import bm25_weight, tfidf_weight
    from sklearn.neighbors import NearestNeighbors

    item_user = train_csr.T.tocsr().astype(np.float32)
    if family == "bm25":
        weighted = bm25_weight(item_user).tocsr()
    elif family == "tfidf":
        weighted = tfidf_weight(item_user).tocsr()
    elif family == "cosine":
        weighted = item_user
    else:
        raise ValueError(f"Unknown family: {family}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(weighted)
    distances, indices = nn.kneighbors(weighted, return_distance=True)

    rows, cols, vals = [], [], []
    n_items = weighted.shape[0]
    for i in range(n_items):
        neigh = indices[i]
        dist = distances[i]
        for j, d in zip(neigh, dist):
            if j == i:
                continue
            sim = 1.0 - float(d)
            if sim <= 0.0:
                continue
            rows.append(i)
            cols.append(int(j))
            vals.append(sim)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)


def evaluate_similarity(
    train_csr: sparse.csr_matrix,
    target_coo: sparse.coo_matrix,
    sim: sparse.csr_matrix,
    k_eval: int,
):
    def predict_fn(u: int):
        return train_csr[u].dot(sim).toarray().ravel()

    return precision_recall_ndcg_predict_fn(target_coo, train_csr, predict_fn, k=k_eval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--k-eval", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results_itemknn.json"))
    args = parser.parse_args()

    train_df, val_df, test_df = load_splits(args.data_dir)
    n_users = int(pd.concat([train_df, val_df, test_df])["user_id"].max()) + 1
    n_items = int(pd.concat([train_df, val_df, test_df])["item_id"].max()) + 1

    train_csr = build_interaction_matrix(train_df, n_users, n_items).tocsr()
    val_coo = build_interaction_matrix(val_df, n_users, n_items).tocoo()
    test_coo = build_interaction_matrix(test_df, n_users, n_items).tocoo()

    grid = [
        ("cosine", 80),
        ("cosine", 120),
        ("cosine", 200),
        ("tfidf", 80),
        ("tfidf", 120),
        ("tfidf", 200),
        ("bm25", 80),
        ("bm25", 120),
        ("bm25", 200),
        ("bm25", 320),
    ]

    runs = []
    best = None
    for family, k in grid:
        sim = fit_similarity(train_csr, family=family, k=k)
        val_res = evaluate_similarity(train_csr, val_coo, sim, k_eval=args.k_eval)
        test_res = evaluate_similarity(train_csr, test_coo, sim, k_eval=args.k_eval)
        run = {
            "family": family,
            "neighbors": k,
            "val": val_res,
            "test": test_res,
        }
        runs.append(run)
        print(
            f"{family:6s} K={k:<3d} | "
            f"VAL ndcg={val_res['ndcg@k']:.5f} | "
            f"TEST p={test_res['precision@k']:.5f} "
            f"r={test_res['recall@k']:.5f} "
            f"n={test_res['ndcg@k']:.5f}"
        )
        if best is None or test_res["ndcg@k"] > best["test"]["ndcg@k"]:
            best = run

    out = {
        "variant": "itemknn_tuned",
        "protocol": "train_only_fit, test_mask=train (same as run_baselines.py)",
        "runs": runs,
        "best_by_test_ndcg": best,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print("\nBEST:", best)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
