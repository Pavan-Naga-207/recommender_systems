from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_ease import main as run_ease_main

RESULTS_DIR = PROJECT_ROOT / "phase2" / "results"
REFRESH_MODEL_RESULTS = False
EXCLUDED_BASELINE_MODELS = {"SVD"}


def load_baseline_results(split: str) -> pd.DataFrame:
    baseline_path = RESULTS_DIR / f"baseline_results_{split}.csv"
    baseline_df = pd.read_csv(baseline_path)
    return baseline_df[~baseline_df["model"].isin(EXCLUDED_BASELINE_MODELS)].reset_index(drop=True)


def combine_results(split: str) -> pd.DataFrame:
    model_path = RESULTS_DIR / f"ease_binary_3000_{split}.csv"
    combined = pd.concat(
        [load_baseline_results(split), pd.read_csv(model_path)],
        ignore_index=True,
    )
    output_path = RESULTS_DIR / f"ease_binary_3000_comparison_{split}.csv"
    combined.to_csv(output_path, index=False)
    return combined


if REFRESH_MODEL_RESULTS:
    run_ease_main()

val_df = combine_results("val")
test_df = combine_results("test")

print("Validation")
print(val_df.to_string(index=False))
print()
print("Test")
print(test_df.to_string(index=False))
print()
print(f"Saved validation comparison to: {RESULTS_DIR / 'ease_binary_3000_comparison_val.csv'}")
print(f"Saved test comparison to: {RESULTS_DIR / 'ease_binary_3000_comparison_test.csv'}")
