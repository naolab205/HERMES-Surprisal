"""Combine the three fixed item-selection runs and paired fold diagnostics."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
parts = []
for tag in ("a_fixed", "b_fixed", "c_fixed"):
    path = ROOT / tag / "tables" / f"charbag_item_selected_{tag}_predictions.csv"
    parts.append(pd.read_csv(path))
candidate = pd.concat(parts, ignore_index=True)
if set(candidate.fold) != set(range(7)) or len(candidate) != 102499:
    raise ValueError("Item-selected predictions do not cover all seven folds")
if candidate.groupby("item_id").fold.nunique().max() != 1:
    raise ValueError("An item appears in multiple folds")
candidate.to_csv(ROOT / "item_selected_predictions.csv", index=False)

baseline = pd.read_csv(ROOT / "baseline_diagnostics" / "extra_metrics_per_fold.csv")
item_selected = pd.read_csv(ROOT / "item_selected_diagnostics" / "extra_metrics_per_fold.csv")
columns = ["trial_spearman", "item_spearman",
           "item_weighted_book_count_spearman", "item_auc_hi_vs_lo"]
paired = baseline[["fold"] + columns].merge(
    item_selected[["fold"] + columns], on="fold", suffixes=("_trial_selected", "_item_selected"),
    validate="one_to_one",
)
for column in columns:
    paired[column + "_delta"] = (
        paired[column + "_item_selected"] - paired[column + "_trial_selected"]
    )
paired.to_csv(ROOT / "paired_fold_metrics.csv", index=False)
summary = {column: {
    "trial_selected_mean": float(paired[column + "_trial_selected"].mean()),
    "item_selected_mean": float(paired[column + "_item_selected"].mean()),
    "paired_delta_mean": float(paired[column + "_delta"].mean()),
    "positive_folds": int((paired[column + "_delta"] > 0).sum()),
} for column in columns}
summary["n_folds"] = 7
(ROOT / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(paired[["fold"] + [column + "_delta" for column in columns]].round(4).to_string(index=False))
print(json.dumps(summary, indent=2))
