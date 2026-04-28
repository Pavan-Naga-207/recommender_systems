from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.common import load_split_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "myket-android-application-market-dataset"
PROCESSED_DIR = PROJECT_ROOT / "phase1" / "processed"
RESULTS_DIR = PROJECT_ROOT / "phase2" / "results"


def load_json(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_stats() -> dict:
    return load_json(PROCESSED_DIR / "dataset_stats.json")


def load_schema() -> dict:
    return load_json(PROCESSED_DIR / "schema.json")


def load_raw_dataset(nrows: int | None = None) -> pd.DataFrame:
    col_names = [
        "user_id",
        "item_id",
        "timestamp",
        "state_label",
        "feature_1",
        "feature_2",
    ]
    return pd.read_csv(
        DATASET_DIR / "myket.csv",
        names=col_names,
        skiprows=1,
        nrows=nrows,
    )


def load_item_metadata() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "app_info_sample.csv")


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, val_df, test_df


def split_time_ranges() -> pd.DataFrame:
    train_df, val_df, test_df = load_processed_splits()
    rows = []
    for split_name, split_df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "min_timestamp": int(split_df["timestamp"].min()),
                "max_timestamp": int(split_df["timestamp"].max()),
                "min_datetime": split_df["timestamp_dt"].min(),
                "max_datetime": split_df["timestamp_dt"].max(),
                "unique_users": split_df["user_id"].nunique(),
                "unique_items": split_df["item_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def eligible_user_summary(eval_split: str = "test", k: int = 10) -> pd.DataFrame:
    split_data = load_split_data(PROCESSED_DIR, eval_split)
    truth_sizes = pd.Series(
        {user_id: len(items) for user_id, items in split_data.eval_user_items.items()},
        name="truth_size",
    )
    train_sizes = pd.Series(
        {user_id: len(items) for user_id, items in split_data.train_user_items.items()},
        name="train_seen",
    )
    eligible = truth_sizes.index.intersection(train_sizes.index)
    eligible_truth = truth_sizes.loc[eligible]
    eligible_train = train_sizes.loc[eligible]
    return pd.DataFrame(
        [
            {
                "eval_split": eval_split,
                "users_in_eval_split": int(truth_sizes.shape[0]),
                "eligible_users": int(len(split_data.eligible_users)),
                "excluded_no_train_history": int(truth_sizes.shape[0] - len(split_data.eligible_users)),
                "users_with_truth_lt_k": int((eligible_truth < k).sum()),
                "min_candidates_after_train_mask": int(split_data.num_items - eligible_train.max()),
                "median_truth_size": float(eligible_truth.median()),
                "max_truth_size": int(eligible_truth.max()),
            }
        ]
    )


def load_result_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_baseline_results(eval_split: str = "test") -> pd.DataFrame:
    return load_result_csv(RESULTS_DIR / f"baseline_results_{eval_split}.csv")


def load_retained_results(eval_split: str = "test") -> pd.DataFrame:
    return load_result_csv(RESULTS_DIR / f"comparison_retained_{eval_split}.csv")


def rank_models(frame: pd.DataFrame, metric: str = "ndcg@10", ascending: bool = False) -> pd.DataFrame:
    ranked = frame.sort_values(metric, ascending=ascending).reset_index(drop=True).copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def metric_delta_table(
    frame: pd.DataFrame,
    reference_model: str = "UserCF",
    metrics: tuple[str, ...] = ("precision@10", "recall@10", "ndcg@10"),
) -> pd.DataFrame:
    if reference_model not in set(frame["model"]):
        raise ValueError(f"Reference model '{reference_model}' not found.")

    ref_row = frame.loc[frame["model"] == reference_model, list(metrics)].iloc[0]
    out = frame[["model", *metrics]].copy()
    for metric in metrics:
        out[f"{metric}_delta_vs_{reference_model}"] = out[metric] - ref_row[metric]
    return out


def plot_metric_bars(frame: pd.DataFrame, metric: str = "ndcg@10", title: str | None = None):
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(metric, ascending=False)
    ax = ordered.plot.bar(x="model", y=metric, legend=False, figsize=(10, 4))
    ax.set_ylabel(metric)
    ax.set_xlabel("model")
    ax.set_title(title or f"Model comparison by {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return ax


def historical_variant_c_results() -> pd.DataFrame:
    rows = [
        {
            "variant_story": "LightFM hybrid with category metadata",
            "precision@10": 0.03638418079096045,
            "recall@10": 0.05123631681254625,
            "ndcg@10": 0.05266477329395565,
        },
        {
            "variant_story": "Retained ItemKNN BM25",
            "precision@10": 0.042994350282485876,
            "recall@10": 0.06350535771287563,
            "ndcg@10": 0.06517995699319355,
        },
    ]
    return pd.DataFrame(rows)
