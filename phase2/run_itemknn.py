from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import build_train_matrix, evaluate_topk, load_split_data, save_results, top_k_from_scores

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")
MODEL_NAME = "ItemKNN-BM25-K320"
NEIGHBORS = 320
TOP_K = 10
SCORE_BATCH_SIZE = 256


def chunked(values: list[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def bm25_weight_rows(matrix: sparse.csr_matrix, k1: float = 100.0, b: float = 0.8) -> sparse.csr_matrix:
    coo = matrix.tocoo(copy=True)
    n_rows = float(coo.shape[0])
    idf = np.log(n_rows) - np.log1p(np.bincount(coo.col, minlength=coo.shape[1]))
    row_sums = np.ravel(coo.sum(axis=1))
    avg_len = row_sums.mean()
    len_norm = (1.0 - b) + b * row_sums / (avg_len + 1e-12)
    coo.data = coo.data * (k1 + 1.0) / (k1 * len_norm[coo.row] + coo.data) * idf[coo.col]
    return coo.tocsr().astype(np.float32)


def fit_itemknn_bm25(train_matrix: sparse.csr_matrix, neighbors: int) -> sparse.csr_matrix:
    item_user = train_matrix.T.tocsr().astype(np.float32)
    weighted = bm25_weight_rows(item_user)

    nn = NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine", algorithm="brute")
    nn.fit(weighted)
    distances, indices = nn.kneighbors(weighted, return_distance=True)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for item_id in range(weighted.shape[0]):
        for neigh_id, dist in zip(indices[item_id], distances[item_id]):
            if neigh_id == item_id:
                continue
            sim = 1.0 - float(dist)
            if sim <= 0.0:
                continue
            rows.append(item_id)
            cols.append(int(neigh_id))
            data.append(sim)

    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(weighted.shape[0], weighted.shape[0]),
        dtype=np.float32,
    )


def run_itemknn_for_split(eval_split: str) -> None:
    split_data = load_split_data(DATA_DIR, eval_split)
    train_matrix = build_train_matrix(
        split_data.train_df,
        split_data.num_users,
        split_data.num_items,
    )
    similarity = fit_itemknn_bm25(train_matrix, NEIGHBORS)

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

    output_csv = OUTPUT_DIR / f"itemknn_bm25_k320_{eval_split}.csv"
    result_df = save_results(results, output_csv, TOP_K, eval_split)
    print(result_df.to_string(index=False))


def main() -> None:
    for eval_split in EVAL_SPLITS:
        print(f"\nRunning ItemKNN-BM25 for: {eval_split}")
        run_itemknn_for_split(eval_split)


if __name__ == "__main__":
    main()
