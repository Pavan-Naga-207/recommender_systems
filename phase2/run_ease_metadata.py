from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
MODEL_PREFIX = "EASEMeta"
BASE_MODEL_NAME = "EASE-Binary-3000"
LAMBDA_REG = 3000.0
TOP_K = 10
SCORE_BATCH_SIZE = 256
METADATA_SIM_TOPK = 200
METADATA_BATCH_SIZE = 512
RERANK_CANDIDATE_SIZE = 200
METADATA_MODES = ("category", "category_word", "category_char")
FUSION_MODES = ("score_blend", "rerank")
ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)


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

    print(f"Fitting {BASE_MODEL_NAME}")
    ease_coeffs = fit_ease(train_matrix, LAMBDA_REG)

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
        return train_matrix[batch_users].dot(ease_coeffs)

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
    val_output = OUTPUT_DIR / "ease_metadata_val_summary.csv"
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
    test_output = OUTPUT_DIR / "ease_metadata_test_best.csv"
    test_df.to_csv(test_output, index=False)
    print("\nBest-config test result")
    print(test_df.to_string(index=False))
    print(f"\nSaved best test result to: {test_output}")


if __name__ == "__main__":
    main()
