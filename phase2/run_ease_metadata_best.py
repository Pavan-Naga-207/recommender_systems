from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import build_train_matrix, evaluate_topk, load_split_data, save_results, top_k_from_scores
from phase2.metadata_utils import build_item_priors

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")
MODEL_NAME = "EASEMeta-ratingPrior-a0.03"
LAMBDA_REG = 3000.0
TOP_K = 10
SCORE_BATCH_SIZE = 256
PRIOR_NAME = "rating"
PRIOR_ALPHA = 0.03


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


def main() -> None:
    val_data = load_split_data(DATA_DIR, "val")
    test_data = load_split_data(DATA_DIR, "test")
    num_users = max(val_data.num_users, test_data.num_users)
    num_items = max(val_data.num_items, test_data.num_items)
    train_matrix = build_train_matrix(val_data.train_df, num_users, num_items)

    ease_coeffs = fit_ease(train_matrix, LAMBDA_REG)
    priors = build_item_priors(num_items)
    item_prior = priors[PRIOR_NAME]

    for eval_split, split_data in (("val", val_data), ("test", test_data)):
        rec_cache: dict[int, list[int]] = {}
        for batch_users in chunked(split_data.eligible_users, SCORE_BATCH_SIZE):
            base_scores = train_matrix[batch_users].dot(ease_coeffs)
            for row_idx, user_id in enumerate(batch_users):
                combined_scores = base_scores[row_idx] + PRIOR_ALPHA * item_prior
                rec_cache[user_id] = top_k_from_scores(
                    combined_scores,
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
        output_csv = OUTPUT_DIR / f"ease_metadata_best_{eval_split}.csv"
        result_df = save_results(results, output_csv, TOP_K, eval_split)
        print(result_df.to_string(index=False))
        print(f"\nSaved metadata-aware EASE results to: {output_csv}\n")


if __name__ == "__main__":
    main()
