from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


INPUT_CSV = "myket-android-application-market-dataset/myket.csv"
OUTPUT_DIR = "phase1/processed"
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

if not (0.0 < TRAIN_RATIO < 1.0):
    raise ValueError("TRAIN_RATIO must be between 0 and 1.")
if not (0.0 <= VAL_RATIO < 1.0):
    raise ValueError("VAL_RATIO must be between 0 and 1.")
if TRAIN_RATIO + VAL_RATIO >= 1.0:
    raise ValueError("TRAIN_RATIO + VAL_RATIO must be < 1.0.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# The raw Myket file has extra zero-valued fields, but the shared pipeline
# only needs the interaction triplet for top-K recommendation.
col_names = [
    "user_id",
    "item_id",
    "timestamp",
    "state_label",
    "feature_1",
    "feature_2",
]
df = pd.read_csv(
    INPUT_CSV,
    names=col_names,
    skiprows=1,
    usecols=["user_id", "item_id", "timestamp"],
)
raw_rows = len(df)

df = df.drop_duplicates()
df = df.dropna(subset=["user_id", "item_id", "timestamp"])
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df["user_id"] = df["user_id"].astype(str)
df["item_id"] = df["item_id"].astype(str)
df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)


# Filtering is repeated because removing low-count users can also make some
# items fall below the item threshold, and vice versa.
while True:
    before = len(df)

    user_counts = df["user_id"].value_counts()
    keep_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    df = df[df["user_id"].isin(keep_users)]

    item_counts = df["item_id"].value_counts()
    keep_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
    df = df[df["item_id"].isin(keep_items)]

    if len(df) == before:
        break

user_codes, user_unique = pd.factorize(df["user_id"], sort=True)
item_codes, item_unique = pd.factorize(df["item_id"], sort=True)
df["user_id"] = user_codes.astype(np.int64)
df["item_id"] = item_codes.astype(np.int64)

df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", origin="unix", errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)


# Global temporal split: train on earlier interactions, validate/test on later
# interactions. This is not a per-user leave-one-out split.
n = len(df)
train_end = int(TRAIN_RATIO * n)
val_end = int((TRAIN_RATIO + VAL_RATIO) * n)
train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("At least one split is empty. Adjust TRAIN_RATIO/VAL_RATIO.")

num_users = int(len(user_unique))
num_items = int(len(item_unique))

# Train-only binary matrix used by the baselines and all phase 2 variants.
interaction_matrix = np.zeros((num_users, num_items), dtype=np.uint8)
train_rows = train_df["user_id"].to_numpy(dtype=np.int64)
train_cols = train_df["item_id"].to_numpy(dtype=np.int64)
interaction_matrix[train_rows, train_cols] = 1

# Save outputs.
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
np.save(os.path.join(OUTPUT_DIR, "interaction_matrix.npy"), interaction_matrix)

pd.DataFrame(
    {
        "original_user_id": user_unique,
        "user_id": np.arange(num_users, dtype=np.int64),
    }
).to_csv(os.path.join(OUTPUT_DIR, "user_mapping.csv"), index=False)

pd.DataFrame(
    {
        "original_item_id": item_unique,
        "item_id": np.arange(num_items, dtype=np.int64),
    }
).to_csv(os.path.join(OUTPUT_DIR, "item_mapping.csv"), index=False)

sparsity = 1.0 - (n / float(num_users * num_items))
stats = {
    "input_csv": INPUT_CSV,
    "raw_rows": int(raw_rows),
    "rows_after_cleaning": int(n),
    "num_users": int(num_users),
    "num_items": int(num_items),
    "sparsity": float(sparsity),
    "split_rows": {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    },
    "filter_thresholds": {
        "min_user_interactions": int(MIN_USER_INTERACTIONS),
        "min_item_interactions": int(MIN_ITEM_INTERACTIONS),
    },
}

with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

schema = {
    "input": {
        "columns": [
            "user_id",
            "item_id",
            "timestamp",
            "state_label",
            "feature_1",
            "feature_2",
        ]
    },
    "processed": {
        "columns": ["user_id", "item_id", "timestamp", "timestamp_dt"]
    },
    "outputs": [
        "train.csv",
        "val.csv",
        "test.csv",
        "interaction_matrix.npy",
        "user_mapping.csv",
        "item_mapping.csv",
        "dataset_stats.json",
        "schema.json",
    ],
}

with open(os.path.join(OUTPUT_DIR, "schema.json"), "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

print("Preprocessing complete.")
print(json.dumps(stats, indent=2))
