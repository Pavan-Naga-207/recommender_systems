from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.sparse import linalg as sparse_linalg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import build_train_matrix, evaluate_topk, load_split_data, save_results, top_k_from_scores

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")
TOP_K = 10
CF_NEIGHBORS = 50
SVD_COMPONENTS = 64


def run_baselines_for_split(eval_split: str) -> None:
    split_data = load_split_data(DATA_DIR, eval_split)
    train_matrix = build_train_matrix(
        split_data.train_df,
        split_data.num_users,
        split_data.num_items,
    )

    # Popularity: same global item order for everyone, with train-seen items skipped.
    item_pop = np.asarray(train_matrix.sum(axis=0)).ravel()
    popularity_ranking = np.argsort(item_pop)[::-1]

    def recommend_popularity(user_id: int, k: int) -> list[int]:
        seen = split_data.train_user_items.get(user_id, set())
        recs = []
        for item_id in popularity_ranking:
            if int(item_id) not in seen:
                recs.append(int(item_id))
            if len(recs) == k:
                break
        return recs

    # UserCF: cosine-style user similarity on binary interaction rows.
    row_norms = np.sqrt(train_matrix.multiply(train_matrix).sum(axis=1)).A1
    row_norms[row_norms == 0.0] = 1.0
    train_matrix_norm = train_matrix.multiply(1.0 / row_norms[:, None]).tocsr()

    def recommend_user_cf(user_id: int, k: int) -> list[int]:
        seen = split_data.train_user_items.get(user_id, set())
        sim = train_matrix_norm[user_id].dot(train_matrix_norm.T).toarray().ravel()
        sim[user_id] = 0.0

        positive = np.where(sim > 0.0)[0]
        if positive.size == 0:
            # If a user has no useful neighbors, fall back to the simple baseline.
            return recommend_popularity(user_id, k)

        n_neighbors = min(CF_NEIGHBORS, positive.size)
        neigh_idx = positive[np.argpartition(sim[positive], -n_neighbors)[-n_neighbors:]]
        neigh_weights = sim[neigh_idx]
        neigh_matrix = train_matrix[neigh_idx].toarray()
        scores = neigh_weights @ neigh_matrix
        return top_k_from_scores(scores, seen, k)

    max_components = min(train_matrix.shape) - 1
    svd_components = max(2, min(SVD_COMPONENTS, max_components))

    # Basic latent-factor baseline from a truncated sparse SVD.
    user_left, singular_vals, item_right_t = sparse_linalg.svds(train_matrix, k=svd_components)
    order = np.argsort(singular_vals)[::-1]
    singular_vals = singular_vals[order]
    user_left = user_left[:, order]
    item_right_t = item_right_t[order, :]
    user_factors = user_left * singular_vals
    item_factors = item_right_t.T

    def recommend_svd(user_id: int, k: int) -> list[int]:
        seen = split_data.train_user_items.get(user_id, set())
        scores = user_factors[user_id] @ item_factors.T
        return top_k_from_scores(scores, seen, k)

    results = [
        evaluate_topk(
            recommend_popularity,
            split_data.eligible_users,
            split_data.eval_user_items,
            TOP_K,
            "Popularity",
        ),
        evaluate_topk(
            recommend_user_cf,
            split_data.eligible_users,
            split_data.eval_user_items,
            TOP_K,
            "UserCF",
        ),
        evaluate_topk(
            recommend_svd,
            split_data.eligible_users,
            split_data.eval_user_items,
            TOP_K,
            "SVD",
        ),
    ]

    output_csv = OUTPUT_DIR / f"baseline_results_{eval_split}.csv"
    result_df = save_results(results, output_csv, TOP_K, eval_split)
    print(result_df.to_string(index=False))
    print(f"\nSaved baseline results to: {output_csv}")


for eval_split in EVAL_SPLITS:
    print(f"\nRunning baselines for: {eval_split}")
    run_baselines_for_split(eval_split)

