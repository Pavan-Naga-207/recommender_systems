from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import build_train_matrix, evaluate_topk, load_split_data, save_results, top_k_from_scores
from phase2.metadata_utils import build_dense_feature_variant, build_item_priors

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")
MODEL_NAME = "RP3betaMetaLowRank-categoryContent"
ALPHA = 0.9
BETA = 0.4
SIMILARITY_TOPK = 400
TOP_K = 10
BLOCK_SIZE = 256
SCORE_BATCH_SIZE = 256
FEATURE_VARIANT = "all_pos_row_norm"
FEATURE_GAMMA = 3.0
CONTENT_VARIANT = "category"
CONTENT_ALPHA = 0.0001
PRIOR_NAME = "rating"
PRIOR_ALPHA = 0.001


def chunked(values: list[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def fit_rp3beta(
    train_matrix: sparse.csr_matrix,
    alpha: float,
    beta: float,
    similarity_topk: int,
    block_size: int,
) -> sparse.csr_matrix:
    p_ui = normalize(train_matrix, norm="l1", axis=1, copy=True)
    p_iu = normalize(train_matrix.T.tocsr(), norm="l1", axis=1, copy=True)

    if alpha != 1.0:
        p_ui.data = np.power(p_ui.data, alpha)
        p_iu.data = np.power(p_iu.data, alpha)

    item_degree = np.ravel(train_matrix.sum(axis=0)).astype(np.float64)
    degree_penalty = np.power(item_degree + 1e-12, -beta)
    degree_penalty[~np.isfinite(degree_penalty)] = 0.0

    num_items = train_matrix.shape[1]
    rows = []
    cols = []
    data = []

    for start in range(0, num_items, block_size):
        end = min(start + block_size, num_items)
        block_scores = (p_iu[start:end] @ p_ui).toarray().astype(np.float32, copy=False)
        block_scores *= degree_penalty[np.newaxis, :]

        for local_row in range(end - start):
            item_id = start + local_row
            row_scores = block_scores[local_row]
            row_scores[item_id] = 0.0

            candidate_idx = np.flatnonzero(row_scores > 0.0)
            if candidate_idx.size == 0:
                continue

            if candidate_idx.size > similarity_topk:
                top_idx = np.argpartition(row_scores[candidate_idx], -similarity_topk)[-similarity_topk:]
                candidate_idx = candidate_idx[top_idx]

            candidate_scores = row_scores[candidate_idx]
            order = np.argsort(candidate_scores)[::-1]
            candidate_idx = candidate_idx[order]
            candidate_scores = candidate_scores[order]

            rows.extend([item_id] * len(candidate_idx))
            cols.extend(candidate_idx.tolist())
            data.extend(candidate_scores.tolist())

    similarity = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(num_items, num_items),
        dtype=np.float32,
    )
    return similarity


def main() -> None:
    val_data = load_split_data(DATA_DIR, "val")
    test_data = load_split_data(DATA_DIR, "test")
    num_users = max(val_data.num_users, test_data.num_users)
    num_items = max(val_data.num_items, test_data.num_items)
    train_matrix = build_train_matrix(val_data.train_df, num_users, num_items).astype(np.float32)

    feature_matrix = build_dense_feature_variant(num_items, FEATURE_VARIANT).astype(np.float32, copy=False)
    content_feature_matrix = build_dense_feature_variant(num_items, CONTENT_VARIANT).astype(np.float32, copy=False)
    user_content_profiles = train_matrix @ content_feature_matrix
    augmented_train_matrix = sparse.vstack(
        [train_matrix, FEATURE_GAMMA * sparse.csr_matrix(feature_matrix.T)],
        format="csr",
        dtype=np.float32,
    )

    rp3beta_similarity = fit_rp3beta(
        train_matrix=augmented_train_matrix,
        alpha=ALPHA,
        beta=BETA,
        similarity_topk=SIMILARITY_TOPK,
        block_size=BLOCK_SIZE,
    )
    priors = build_item_priors(num_items)
    item_prior = priors[PRIOR_NAME]

    for eval_split, split_data in (("val", val_data), ("test", test_data)):
        rec_cache: dict[int, list[int]] = {}
        for batch_users in chunked(split_data.eligible_users, SCORE_BATCH_SIZE):
            base_scores = train_matrix[batch_users].dot(rp3beta_similarity).toarray()
            content_scores = user_content_profiles[batch_users] @ content_feature_matrix.T
            combined_scores = base_scores + CONTENT_ALPHA * content_scores + PRIOR_ALPHA * item_prior
            for row_idx, user_id in enumerate(batch_users):
                rec_cache[user_id] = top_k_from_scores(
                    combined_scores[row_idx],
                    split_data.train_user_items.get(user_id, set()),
                    TOP_K,
                )

        results = [
            evaluate_topk(
                lambda user_id, k: rec_cache[user_id][:k],
                split_data.eligible_users,
                split_data.eval_user_items,
                TOP_K,
                MODEL_NAME,
            )
        ]
        output_csv = OUTPUT_DIR / f"rp3beta_metadata_best_{eval_split}.csv"
        result_df = save_results(results, output_csv, TOP_K, eval_split)
        print(result_df.to_string(index=False))
        print(f"\nSaved metadata-aware RP3beta results to: {output_csv}\n")


if __name__ == "__main__":
    main()
