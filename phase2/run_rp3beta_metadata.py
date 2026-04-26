from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import build_train_matrix, load_split_data
from phase2.metadata_utils import (
    HybridConfig,
    build_metadata_similarity,
    build_recommendation_cache,
    evaluate_config_results,
)

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
MODEL_PREFIX = "RP3betaMeta"
BASE_MODEL_NAME = "RP3beta-a0.9-b0.4-t400"
ALPHA = 0.9
BETA = 0.4
SIMILARITY_TOPK = 400
TOP_K = 10
BLOCK_SIZE = 256
SCORE_BATCH_SIZE = 256
METADATA_SIM_TOPK = 200
METADATA_BATCH_SIZE = 512
RERANK_CANDIDATE_SIZE = 200
METADATA_MODES = ("category", "category_word", "category_char")
FUSION_MODES = ("score_blend", "rerank")
ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)


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
    train_matrix = build_train_matrix(val_data.train_df, num_users, num_items)

    print(f"Fitting {BASE_MODEL_NAME}")
    rp3beta_similarity = fit_rp3beta(
        train_matrix=train_matrix,
        alpha=ALPHA,
        beta=BETA,
        similarity_topk=SIMILARITY_TOPK,
        block_size=BLOCK_SIZE,
    )

    print("Building metadata similarities")
    metadata_similarity_by_mode = {
        mode: build_metadata_similarity(num_items, mode, METADATA_SIM_TOPK, METADATA_BATCH_SIZE)
        for mode in METADATA_MODES
    }

    val_configs = [
        HybridConfig(metadata_mode=mode, fusion=fusion, alpha=alpha)
        for mode in METADATA_MODES
        for fusion in FUSION_MODES
        for alpha in ALPHAS
    ]

    def base_score_fn(batch_users: list[int]) -> np.ndarray:
        return train_matrix[batch_users].dot(rp3beta_similarity).toarray()

    print("Running validation sweep")
    val_rec_cache = build_recommendation_cache(
        train_matrix=train_matrix,
        eligible_users=val_data.eligible_users,
        train_user_items=val_data.train_user_items,
        base_score_fn=base_score_fn,
        metadata_similarity_by_mode=metadata_similarity_by_mode,
        configs=val_configs,
        top_k=TOP_K,
        score_batch_size=SCORE_BATCH_SIZE,
        rerank_candidate_size=RERANK_CANDIDATE_SIZE,
    )
    val_df = evaluate_config_results(
        MODEL_PREFIX,
        "val",
        val_data,
        val_configs,
        val_rec_cache,
        TOP_K,
    )
    val_output = OUTPUT_DIR / "rp3beta_metadata_val_summary.csv"
    val_df.to_csv(val_output, index=False)
    print(val_df.head(10).to_string(index=False))
    print(f"\nSaved validation sweep to: {val_output}")

    best_row = val_df.iloc[0]
    best_config = HybridConfig(
        metadata_mode=str(best_row["metadata_mode"]),
        fusion=str(best_row["fusion"]),
        alpha=float(best_row["alpha"]),
    )
    print(f"\nBest validation config: {best_config}")

    test_rec_cache = build_recommendation_cache(
        train_matrix=train_matrix,
        eligible_users=test_data.eligible_users,
        train_user_items=test_data.train_user_items,
        base_score_fn=base_score_fn,
        metadata_similarity_by_mode=metadata_similarity_by_mode,
        configs=[best_config],
        top_k=TOP_K,
        score_batch_size=SCORE_BATCH_SIZE,
        rerank_candidate_size=RERANK_CANDIDATE_SIZE,
    )
    test_df = evaluate_config_results(
        MODEL_PREFIX,
        "test",
        test_data,
        [best_config],
        test_rec_cache,
        TOP_K,
    )
    test_output = OUTPUT_DIR / "rp3beta_metadata_test_best.csv"
    test_df.to_csv(test_output, index=False)
    print("\nBest-config test result")
    print(test_df.to_string(index=False))
    print(f"\nSaved best test result to: {test_output}")


if __name__ == "__main__":
    main()
