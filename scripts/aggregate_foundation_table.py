#!/usr/bin/env python
"""Assemble the foundation ablation table against the frozen JBHI baseline.

The frozen row is copied from the sealed package's own summary CSV rather than
recomputed, so the comparison is against the number that was actually reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "tables" / "surprisal_cv_full_summary.csv"
FROZEN_PER_FOLD = ROOT / "results" / "tables" / "surprisal_cv_full_per_fold.csv"
METRICS = ("trial_spearman", "item_spearman", "item_auc_hi_vs_lo")


def frozen_baseline() -> tuple[dict, pd.DataFrame]:
    summary = pd.read_csv(FROZEN)
    eeg = summary[summary["model"] == "CogAlign_eeg_only"].iloc[0]
    row = {
        "arm": "frozen_row_resampled (EEG-only)",
        "trial_spearman_mean": float(eeg["trial_rho_mean"]),
        "trial_spearman_std": float(eeg["trial_rho_std"]),
        "item_spearman_mean": float(eeg["item_rho_mean"]),
        "item_spearman_std": float(eeg["item_rho_std"]),
        "item_auc_hi_vs_lo_mean": float(eeg["item_auc_mean"]),
        "item_auc_hi_vs_lo_std": float(eeg["item_auc_std"]),
    }
    per_fold = pd.read_csv(FROZEN_PER_FOLD)
    per_fold = per_fold[per_fold["model"] == "CogAlign_eeg_only"].copy()
    per_fold = per_fold.rename(columns={"item_auc_hi_vs_lo": "item_auc_hi_vs_lo"})
    return row, per_fold[["fold", "trial_spearman", "item_spearman", "item_auc_hi_vs_lo"]]


def main() -> None:
    tables = ROOT / "results" / "tables"
    base_row, base_folds = frozen_baseline()
    rows = [base_row]
    paired = []

    for path in sorted(tables.glob("charbag_*_summary.json")):
        arm = json.loads(path.read_text())
        if arm.get("n_folds", 0) < 1:
            continue
        rows.append({"arm": arm["arm"], **{k: v for k, v in arm.items() if k.endswith(("_mean", "_std"))}})

        fold_path = tables / f"charbag_{arm['arm']}_per_fold.csv"
        if not fold_path.exists():
            continue
        folds = pd.read_csv(fold_path)
        merged = base_folds.merge(folds, on="fold", suffixes=("_frozen", "_new"))
        if len(merged) < 3:
            continue
        entry = {"arm": arm["arm"], "n_paired_folds": len(merged)}
        for metric in METRICS:
            delta = merged[f"{metric}_new"] - merged[f"{metric}_frozen"]
            entry[f"{metric}_delta"] = float(delta.mean())
            entry[f"{metric}_rel_pct"] = float(
                100.0 * delta.mean() / abs(merged[f"{metric}_frozen"].mean())
            )
            # Seven paired folds is few, so report the exact signed-rank p and
            # the win count rather than leaning on the p-value alone.
            entry[f"{metric}_folds_improved"] = int((delta > 0).sum())
            try:
                entry[f"{metric}_wilcoxon_p"] = float(wilcoxon(delta).pvalue)
            except ValueError:
                entry[f"{metric}_wilcoxon_p"] = float("nan")
        paired.append(entry)

    table = pd.DataFrame(rows)
    table.to_csv(tables / "foundation_comparison.csv", index=False)
    print("\n=== Arm summary (7-fold mean +/- sd) ===")
    for _, r in table.iterrows():
        print(
            f"{r['arm']:<36s} trial {r['trial_spearman_mean']:.4f}+/-{r['trial_spearman_std']:.4f}"
            f"  item {r['item_spearman_mean']:.4f}+/-{r['item_spearman_std']:.4f}"
            f"  auc {r['item_auc_hi_vs_lo_mean']:.4f}+/-{r['item_auc_hi_vs_lo_std']:.4f}"
        )

    if paired:
        delta_table = pd.DataFrame(paired)
        delta_table.to_csv(tables / "foundation_paired_deltas.csv", index=False)
        print("\n=== Paired per-fold change vs the frozen baseline ===")
        for _, r in delta_table.iterrows():
            print(f"\n{r['arm']}  (n={int(r['n_paired_folds'])} folds)")
            for metric in METRICS:
                print(
                    f"  {metric:<20s} {r[f'{metric}_delta']:+.4f} "
                    f"({r[f'{metric}_rel_pct']:+.1f}%)  "
                    f"folds improved {int(r[f'{metric}_folds_improved'])}/{int(r['n_paired_folds'])}  "
                    f"p={r[f'{metric}_wilcoxon_p']:.4f}"
                )


if __name__ == "__main__":
    main()
