from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DEFAULT_DATA_DIR = Path("phase1/processed")
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class LoadedSplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    eval_df: pd.DataFrame
    num_users: int
    num_items: int
    train_user_items: dict[int, set[int]]
    eval_user_items: dict[int, set[int]]
    eligible_users: list[int]


def _group_items_by_user(df: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(user_id): set(items.astype(int))
        for user_id, items in df.groupby("user_id")["item_id"]
    }


def load_split_data(data_dir: str | Path = DEFAULT_DATA_DIR, eval_split: str = "test") -> LoadedSplitData:
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be either 'val' or 'test'.")

    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    eval_df = val_df if eval_split == "val" else test_df

    num_users = int(max(train_df["user_id"].max(), eval_df["user_id"].max()) + 1)
    num_items = int(max(train_df["item_id"].max(), eval_df["item_id"].max()) + 1)

    train_user_items = _group_items_by_user(train_df)
    eval_user_items = _group_items_by_user(eval_df)
    eligible_users = sorted(set(eval_user_items) & set(train_user_items))
    if not eligible_users:
        raise ValueError("No users overlap between train split and evaluation split.")

    return LoadedSplitData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        eval_df=eval_df,
        num_users=num_users,
        num_items=num_items,
        train_user_items=train_user_items,
        eval_user_items=eval_user_items,
        eligible_users=eligible_users,
    )


def build_train_matrix(train_df: pd.DataFrame, num_users: int, num_items: int) -> sparse.csr_matrix:
    train_rows = train_df["user_id"].to_numpy(dtype=np.int64)
    train_cols = train_df["item_id"].to_numpy(dtype=np.int64)
    train_data = np.ones(len(train_df), dtype=np.float32)
    train_matrix = sparse.csr_matrix(
        (train_data, (train_rows, train_cols)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )
    train_matrix.data[:] = 1.0
    return train_matrix


def top_k_from_scores(scores: np.ndarray, seen_items: set[int], k: int) -> list[int]:
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


def evaluate_topk(
    recommender,
    users: list[int],
    ground_truth: dict[int, set[int]],
    k: int,
    model_name: str,
) -> dict[str, float | int | str]:
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


def save_results(
    results: list[dict[str, float | int | str]],
    output_csv: str | Path,
    top_k: int,
    eval_split: str,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(
        [
            {
                "model": result["model"],
                f"precision@{top_k}": result["precision_at_k"],
                f"recall@{top_k}": result["recall_at_k"],
                f"ndcg@{top_k}": result["ndcg_at_k"],
                "users_evaluated": result["users_evaluated"],
                "eval_split": eval_split,
            }
            for result in results
        ]
    )
    result_df.to_csv(output_csv, index=False)
    return result_df
