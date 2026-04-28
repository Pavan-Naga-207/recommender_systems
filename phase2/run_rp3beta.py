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

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")
MODEL_NAME = "RP3beta-a0.9-b0.4-t400"
ALPHA = 0.9
BETA = 0.4
SIMILARITY_TOPK = 400
TOP_K = 10
BLOCK_SIZE = 256
SCORE_BATCH_SIZE = 256


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
    # Normalize the bipartite graph into user->item and item->user transition matrices.
    p_ui = normalize(train_matrix, norm="l1", axis=1, copy=True)
    p_iu = normalize(train_matrix.T.tocsr(), norm="l1", axis=1, copy=True)

    if alpha != 1.0:
        p_ui.data = np.power(p_ui.data, alpha)
        p_iu.data = np.power(p_iu.data, alpha)

    item_degree = np.ravel(train_matrix.sum(axis=0)).astype(np.float64)
    # Beta downweights very popular items so the walk does not collapse into popularity.
    degree_penalty = np.power(item_degree + 1e-12, -beta)
    degree_penalty[~np.isfinite(degree_penalty)] = 0.0

    num_items = train_matrix.shape[1]
    rows = []
    cols = []
    data = []

    for start in range(0, num_items, block_size):
        end = min(start + block_size, num_items)
        # Item -> user -> item random-walk scores for this item block.
        block_scores = (p_iu[start:end] @ p_ui).toarray().astype(np.float32, copy=False)
        block_scores *= degree_penalty[np.newaxis, :]

        for local_row in range(end - start):
            item_id = start + local_row
            row_scores = block_scores[local_row]
            row_scores[item_id] = 0.0

            positive_mask = row_scores > 0.0
            if not np.any(positive_mask):
                continue

            candidate_idx = np.flatnonzero(positive_mask)
            if candidate_idx.size > similarity_topk:
                # Keep only the strongest neighbors so scoring stays sparse enough.
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


def run_rp3beta_for_split(eval_split: str) -> None:
    split_data = load_split_data(DATA_DIR, eval_split)
    train_matrix = build_train_matrix(
        split_data.train_df,
        split_data.num_users,
        split_data.num_items,
    )
    similarity = fit_rp3beta(
        train_matrix=train_matrix,
        alpha=ALPHA,
        beta=BETA,
        similarity_topk=SIMILARITY_TOPK,
        block_size=BLOCK_SIZE,
    )

    rec_cache: dict[int, list[int]] = {}
    for batch_users in chunked(split_data.eligible_users, SCORE_BATCH_SIZE):
        batch_scores = train_matrix[batch_users].dot(similarity).toarray()
        for row_idx, user_id in enumerate(batch_users):
            rec_cache[user_id] = top_k_from_scores(
                batch_scores[row_idx],
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

    output_csv = OUTPUT_DIR / f"rp3beta_a0.9_b0.4_t400_{eval_split}.csv"
    result_df = save_results(results, output_csv, TOP_K, eval_split)
    print(result_df.to_string(index=False))


def main() -> None:
    for eval_split in EVAL_SPLITS:
        print(f"\nRunning RP3beta for: {eval_split}")
        run_rp3beta_for_split(eval_split)


if __name__ == "__main__":
    main()

