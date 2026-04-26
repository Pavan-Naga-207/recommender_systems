from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.common import evaluate_topk, top_k_from_scores

DATA_DIR = PROJECT_ROOT / "phase1" / "processed"
APP_INFO_PATH = PROJECT_ROOT / "myket-android-application-market-dataset" / "app_info_sample.csv"
DENSE_FEATURE_PATH = PROJECT_ROOT / "myket-android-application-market-dataset" / "data_int_index" / "app_info_sample.npy"
APP_NAME_MAPPING_PATH = PROJECT_ROOT / "myket-android-application-market-dataset" / "data_int_index" / "app_name_mapping.csv"
CACHE_DIR = PROJECT_ROOT / "phase2" / "results" / "cache"

PACKAGE_STOPWORDS = {
    "com",
    "ir",
    "net",
    "org",
    "co",
    "app",
    "apps",
    "android",
    "mobile",
    "market",
    "myket",
    "air",
}


@dataclass(frozen=True)
class HybridConfig:
    fusion: str
    metadata_mode: str
    alpha: float

    @property
    def key(self) -> str:
        alpha_text = str(self.alpha).replace(".", "p")
        return f"{self.metadata_mode}__{self.fusion}__a{alpha_text}"


def chunked(values: list[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def load_item_metadata(num_items: int) -> pd.DataFrame:
    item_map = pd.read_csv(DATA_DIR / "item_mapping.csv")
    meta = pd.read_csv(APP_INFO_PATH)
    merged = item_map.merge(
        meta[["app_name", "category_en", "installs", "rating", "rating_count"]],
        left_on="original_item_id",
        right_on="app_name",
        how="left",
    )
    merged = merged.sort_values("item_id").drop_duplicates("item_id").reset_index(drop=True)
    if len(merged) != num_items:
        item_ids = np.arange(num_items, dtype=np.int64)
        merged = pd.DataFrame({"item_id": item_ids}).merge(merged, on="item_id", how="left")
    return merged


def build_item_priors(num_items: int) -> dict[str, np.ndarray]:
    metadata_df = load_item_metadata(num_items)

    installs = np.log1p(metadata_df["installs"].fillna(0.0).to_numpy(dtype=np.float64))
    installs = (installs - installs.mean()) / (installs.std() + 1e-12)

    rating_count = np.log1p(metadata_df["rating_count"].fillna(0.0).to_numpy(dtype=np.float64))
    rating_count = (rating_count - rating_count.mean()) / (rating_count.std() + 1e-12)

    rating = metadata_df["rating"].fillna(0.0).to_numpy(dtype=np.float64)
    rating = (rating - rating.mean()) / (rating.std() + 1e-12)

    return {
        "installs": installs.astype(np.float32),
        "rating_count": rating_count.astype(np.float32),
        "rating": rating.astype(np.float32),
        "pop": (0.6 * installs + 0.4 * rating_count).astype(np.float32),
        "all": (0.45 * installs + 0.35 * rating_count + 0.2 * rating).astype(np.float32),
    }


def load_dense_item_features(num_items: int) -> np.ndarray:
    item_map = pd.read_csv(DATA_DIR / "item_mapping.csv")
    app_name_map = pd.read_csv(APP_NAME_MAPPING_PATH)
    dense_features = np.load(DENSE_FEATURE_PATH).astype(np.float32, copy=False)
    dense_features = np.nan_to_num(dense_features, nan=0.0, posinf=0.0, neginf=0.0)

    merged = item_map.merge(
        app_name_map,
        left_on="original_item_id",
        right_on="app_name",
        how="left",
    )
    merged = merged.sort_values("item_id").drop_duplicates("item_id").reset_index(drop=True)
    if len(merged) != num_items:
        item_ids = np.arange(num_items, dtype=np.int64)
        merged = pd.DataFrame({"item_id": item_ids}).merge(merged, on="item_id", how="left")

    features = np.zeros((num_items, dense_features.shape[1]), dtype=np.float32)
    valid = merged["app_name_id"].notna()
    if valid.any():
        rows = merged.loc[valid, "item_id"].to_numpy(dtype=np.int64)
        feature_rows = merged.loc[valid, "app_name_id"].to_numpy(dtype=np.int64)
        features[rows] = dense_features[feature_rows]
    return features


def build_dense_feature_variant(num_items: int, variant: str) -> np.ndarray:
    features = load_dense_item_features(num_items)
    numeric = features[:, :3].astype(np.float32, copy=True)
    categories = features[:, 3:].astype(np.float32, copy=False)
    numeric_positive = numeric.copy()

    numeric[:, 0] = np.log1p(np.clip(numeric[:, 0], 0.0, None))
    numeric[:, 2] = np.log1p(np.clip(numeric[:, 2], 0.0, None))
    numeric_mean = numeric.mean(axis=0, keepdims=True)
    numeric_std = numeric.std(axis=0, keepdims=True) + 1e-12
    numeric = (numeric - numeric_mean) / numeric_std
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    categories = np.nan_to_num(categories, nan=0.0, posinf=0.0, neginf=0.0)

    numeric_positive[:, 0] = np.log1p(np.clip(numeric_positive[:, 0], 0.0, None))
    numeric_positive[:, 2] = np.log1p(np.clip(numeric_positive[:, 2], 0.0, None))
    numeric_min = numeric_positive.min(axis=0, keepdims=True)
    numeric_max = numeric_positive.max(axis=0, keepdims=True)
    numeric_positive = (numeric_positive - numeric_min) / (numeric_max - numeric_min + 1e-12)
    numeric_positive = np.nan_to_num(numeric_positive, nan=0.0, posinf=0.0, neginf=0.0)

    if variant == "category":
        feature_matrix = categories
    elif variant == "numeric_z":
        feature_matrix = numeric
    elif variant == "all_z":
        feature_matrix = np.concatenate([numeric, categories], axis=1)
    elif variant == "all_row_norm":
        feature_matrix = np.concatenate([numeric, categories], axis=1)
        row_norm = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-12
        feature_matrix = feature_matrix / row_norm
    elif variant == "all_pos_row_norm":
        feature_matrix = np.concatenate([numeric_positive, categories], axis=1)
        row_norm = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-12
        feature_matrix = feature_matrix / row_norm
    else:
        raise ValueError(f"Unsupported dense feature variant: {variant}")

    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_matrix.astype(np.float32, copy=False)


def tokenize_package_name(value: str | float) -> str:
    if pd.isna(value):
        return ""
    tokens = re.split(r"[^a-zA-Z0-9]+", str(value).lower())
    filtered = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in PACKAGE_STOPWORDS:
            continue
        filtered.append(token)
    return " ".join(filtered)


def build_category_matrix(metadata_df: pd.DataFrame, num_items: int) -> sparse.csr_matrix:
    valid = metadata_df["category_en"].notna()
    if not valid.any():
        return sparse.csr_matrix((num_items, 0), dtype=np.float32)

    categories = pd.Categorical(metadata_df.loc[valid, "category_en"])
    rows = metadata_df.loc[valid, "item_id"].to_numpy(dtype=np.int64)
    cols = categories.codes.astype(np.int64, copy=False)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_items, len(categories.categories)),
        dtype=np.float32,
    )


def build_word_token_matrix(metadata_df: pd.DataFrame, num_items: int) -> sparse.csr_matrix:
    docs = [""] * num_items
    for row in metadata_df.itertuples(index=False):
        docs[int(row.item_id)] = tokenize_package_name(row.original_item_id)

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.2,
        sublinear_tf=True,
        norm="l2",
    )
    token_matrix = vectorizer.fit_transform(docs)
    return token_matrix.astype(np.float32, copy=False).tocsr()


def build_char_ngram_matrix(metadata_df: pd.DataFrame, num_items: int) -> sparse.csr_matrix:
    docs = [""] * num_items
    for row in metadata_df.itertuples(index=False):
        docs[int(row.item_id)] = str(row.original_item_id or "")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.2,
        sublinear_tf=True,
        norm="l2",
    )
    token_matrix = vectorizer.fit_transform(docs)
    return token_matrix.astype(np.float32, copy=False).tocsr()


def build_metadata_feature_matrix(metadata_df: pd.DataFrame, num_items: int, mode: str) -> sparse.csr_matrix:
    category_matrix = build_category_matrix(metadata_df, num_items)

    if mode == "category":
        features = category_matrix
    elif mode == "category_word":
        word_matrix = build_word_token_matrix(metadata_df, num_items)
        features = sparse.hstack([category_matrix, word_matrix], format="csr")
    elif mode == "category_char":
        char_matrix = build_char_ngram_matrix(metadata_df, num_items)
        features = sparse.hstack([category_matrix, char_matrix], format="csr")
    else:
        raise ValueError(f"Unsupported metadata mode: {mode}")

    return normalize(features, norm="l2", axis=1, copy=False).tocsr()


def keep_topk_sparse_rows(similarity: sparse.csr_matrix, topk: int) -> sparse.csr_matrix:
    similarity = similarity.tolil(copy=False)
    for row_idx in range(similarity.shape[0]):
        row_data = np.asarray(similarity.data[row_idx], dtype=np.float32)
        row_cols = np.asarray(similarity.rows[row_idx], dtype=np.int64)
        if row_data.size <= topk:
            continue

        top_idx = np.argpartition(row_data, -topk)[-topk:]
        order = np.argsort(row_data[top_idx])[::-1]
        keep_idx = top_idx[order]
        similarity.rows[row_idx] = row_cols[keep_idx].tolist()
        similarity.data[row_idx] = row_data[keep_idx].tolist()

    similarity = similarity.tocsr()
    return normalize(similarity, norm="l1", axis=1, copy=False).tocsr()


def build_metadata_similarity(
    num_items: int,
    mode: str,
    topk: int,
    batch_size: int,
) -> sparse.csr_matrix:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"metadata_similarity_{mode}_k{topk}.npz"
    if cache_path.exists():
        return load_npz(cache_path).tocsr()

    metadata_df = load_item_metadata(num_items)
    features = build_metadata_feature_matrix(metadata_df, num_items, mode)

    rows = []
    cols = []
    data = []
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        block = features[start:end] @ features.T
        block = block.toarray().astype(np.float32, copy=False)
        for local_row in range(end - start):
            item_id = start + local_row
            row_scores = block[local_row]
            row_scores[item_id] = 0.0
            positive_idx = np.flatnonzero(row_scores > 0.0)
            if positive_idx.size == 0:
                continue
            if positive_idx.size > topk:
                top_idx = np.argpartition(row_scores[positive_idx], -topk)[-topk:]
                positive_idx = positive_idx[top_idx]
            candidate_scores = row_scores[positive_idx]
            order = np.argsort(candidate_scores)[::-1]
            positive_idx = positive_idx[order]
            candidate_scores = candidate_scores[order]
            rows.extend([item_id] * len(positive_idx))
            cols.extend(positive_idx.tolist())
            data.extend(candidate_scores.tolist())

    similarity = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(num_items, num_items),
        dtype=np.float32,
    )
    similarity = normalize(similarity, norm="l1", axis=1, copy=False).tocsr()
    save_npz(cache_path, similarity)
    return similarity


def select_top_k_reranked(
    base_scores: np.ndarray,
    metadata_scores: np.ndarray,
    seen_items: set[int],
    top_k: int,
    candidate_size: int,
    alpha: float,
) -> list[int]:
    scores = np.asarray(base_scores, dtype=np.float64).copy()
    if seen_items:
        seen_idx = np.fromiter(seen_items, dtype=np.int64)
        scores[seen_idx] = -np.inf

    finite_mask = np.isfinite(scores)
    candidate_count = int(finite_mask.sum())
    if candidate_count == 0:
        return []

    candidate_eff = min(candidate_size, candidate_count)
    candidate_idx = np.argpartition(scores, -candidate_eff)[-candidate_eff:]
    rerank_scores = scores[candidate_idx] + alpha * metadata_scores[candidate_idx]
    order = np.argsort(rerank_scores)[::-1]
    return candidate_idx[order[: min(top_k, candidate_eff)]].tolist()


def build_recommendation_cache(
    train_matrix: sparse.csr_matrix,
    eligible_users: list[int],
    train_user_items: dict[int, set[int]],
    base_score_fn,
    metadata_similarity_by_mode: dict[str, sparse.csr_matrix],
    configs: list[HybridConfig],
    top_k: int,
    score_batch_size: int,
    rerank_candidate_size: int,
) -> dict[str, dict[int, list[int]]]:
    rec_cache_by_config = {config.key: {} for config in configs}

    for batch_users in chunked(eligible_users, score_batch_size):
        base_batch_scores = base_score_fn(batch_users)
        metadata_batch_scores = {
            mode: train_matrix[batch_users].dot(similarity).toarray()
            for mode, similarity in metadata_similarity_by_mode.items()
        }

        for row_idx, user_id in enumerate(batch_users):
            seen_items = train_user_items.get(user_id, set())
            base_scores = base_batch_scores[row_idx]
            for config in configs:
                metadata_scores = metadata_batch_scores[config.metadata_mode][row_idx]
                if config.fusion == "score_blend":
                    combined_scores = base_scores + config.alpha * metadata_scores
                    recs = top_k_from_scores(combined_scores, seen_items, top_k)
                elif config.fusion == "rerank":
                    recs = select_top_k_reranked(
                        base_scores,
                        metadata_scores,
                        seen_items,
                        top_k,
                        rerank_candidate_size,
                        config.alpha,
                    )
                else:
                    raise ValueError(f"Unsupported fusion mode: {config.fusion}")
                rec_cache_by_config[config.key][user_id] = recs

    return rec_cache_by_config


def evaluate_config_results(
    model_prefix: str,
    split_name: str,
    split_data,
    configs: list[HybridConfig],
    rec_cache_by_config: dict[str, dict[int, list[int]]],
    top_k: int,
) -> pd.DataFrame:
    results = []
    for config in configs:
        model_name = f"{model_prefix}-{config.metadata_mode}-{config.fusion}-a{config.alpha:g}"
        metrics = evaluate_topk(
            lambda user_id, k, key=config.key: rec_cache_by_config[key][user_id][:k],
            split_data.eligible_users,
            split_data.eval_user_items,
            top_k,
            model_name,
        )
        results.append(
            {
                "model": model_name,
                "metadata_mode": config.metadata_mode,
                "fusion": config.fusion,
                "alpha": config.alpha,
                f"precision@{top_k}": metrics["precision_at_k"],
                f"recall@{top_k}": metrics["recall_at_k"],
                f"ndcg@{top_k}": metrics["ndcg_at_k"],
                "users_evaluated": metrics["users_evaluated"],
                "eval_split": split_name,
            }
        )

    result_df = pd.DataFrame(results).sort_values(
        by=[f"ndcg@{top_k}", f"precision@{top_k}", f"recall@{top_k}"],
        ascending=False,
    )
    return result_df.reset_index(drop=True)
