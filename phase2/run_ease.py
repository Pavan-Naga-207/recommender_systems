from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import build_train_matrix, evaluate_topk, load_split_data, save_results, top_k_from_scores

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")
MODEL_NAME = "EASE-Binary-3000"
LAMBDA_REG = 3000.0
TOP_K = 10
SCORE_BATCH_SIZE = 256


def chunked(values: list[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def fit_ease(interaction_matrix, lambda_reg: float) -> np.ndarray:
    gram = (interaction_matrix.T @ interaction_matrix).toarray().astype(np.float64, copy=False)
    diag_idx = np.diag_indices_from(gram)
    gram[diag_idx] += lambda_reg
    precision = np.linalg.inv(gram)
    coeffs = -precision / np.diag(precision)
    coeffs[diag_idx] = 0.0
    return coeffs.astype(np.float32, copy=False)


def run_ease_for_split(eval_split: str) -> None:
    split_data = load_split_data(DATA_DIR, eval_split)
    train_matrix = build_train_matrix(
        split_data.train_df,
        split_data.num_users,
        split_data.num_items,
    )
    coeffs = fit_ease(train_matrix, LAMBDA_REG)

    rec_cache: dict[int, list[int]] = {}
    for batch_users in chunked(split_data.eligible_users, SCORE_BATCH_SIZE):
        batch_scores = train_matrix[batch_users].dot(coeffs)
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

    output_csv = OUTPUT_DIR / f"ease_binary_3000_{eval_split}.csv"
    result_df = save_results(results, output_csv, TOP_K, eval_split)
    print(result_df.to_string(index=False))


for eval_split in EVAL_SPLITS:
    print(f"\nRunning EASE for: {eval_split}")
    run_ease_for_split(eval_split)

