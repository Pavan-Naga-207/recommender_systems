from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "phase2" / "results"
EVAL_SPLITS = ("val", "test")


def combine_results_for_split(eval_split: str) -> None:
    baseline_path = RESULTS_DIR / f"baseline_results_{eval_split}.csv"
    result_paths = [
        RESULTS_DIR / f"ease_binary_3000_{eval_split}.csv",
        RESULTS_DIR / f"rp3beta_a0.9_b0.4_t400_{eval_split}.csv",
    ]

    frames = [pd.read_csv(baseline_path)]
    for path in result_paths:
        frames.append(pd.read_csv(path))

    combined = pd.concat(frames, ignore_index=True)
    output_csv = RESULTS_DIR / f"comparison_retained_{eval_split}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(combined.to_string(index=False))


for eval_split in EVAL_SPLITS:
    print(f"\nCombining results for: {eval_split}")
    combine_results_for_split(eval_split)
