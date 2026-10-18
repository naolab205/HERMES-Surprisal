#!/usr/bin/env python3
"""Summarize the matched seven-fold ocular-correction experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/eye_clean_decoder_20260930"
TABLES = OUT / "tables"
METRICS = ("trial_spearman", "item_spearman", "item_auc_hi_vs_lo")


def read(name: str) -> pd.DataFrame:
    path = TABLES / f"charbag_{name}_per_fold.csv"
    frame = pd.read_csv(path).set_index("fold")
    return frame.loc[list(range(7)), list(METRICS)]


def main() -> None:
    baseline_path = ROOT / "results/tables/charbag_ABL_noeye_per_fold.csv"
    archived_baseline = pd.read_csv(baseline_path).set_index("fold").loc[list(range(7)), list(METRICS)]
    baseline = pd.concat((
        pd.read_csv(TABLES / "charbag_paired_eeg_only_per_fold.csv").set_index("fold"),
        pd.read_csv(TABLES / "charbag_paired_eeg_only_rest_per_fold.csv").set_index("fold"),
    )).sort_index().loc[list(range(7)), list(METRICS)]
    pilot = pd.read_csv(TABLES / "charbag_paired_eye_clean_per_fold.csv").set_index("fold")
    rest = pd.read_csv(TABLES / "charbag_paired_eye_clean_rest_per_fold.csv").set_index("fold")
    clean = pd.concat((pilot, rest)).sort_index().loc[list(range(7)), list(METRICS)]
    fusion = read("paired_eye_fusion")
    max_archive_difference = float((baseline - archived_baseline).abs().max().max())
    if max_archive_difference > 1e-6:
        raise RuntimeError(f"Fresh EEG-only run differs from archive: {max_archive_difference}")

    table = pd.concat({"eeg_only": baseline, "eye_clean_eeg_only": clean,
                       "eeg_eye_fusion": fusion}, axis=1)
    table.columns = [f"{arm}_{metric}" for arm, metric in table.columns]
    for arm, frame in (("eye_clean_eeg_only", clean), ("eeg_eye_fusion", fusion)):
        for metric in METRICS:
            table[f"delta_{arm}_vs_eeg_only_{metric}"] = frame[metric] - baseline[metric]
    table.to_csv(OUT / "paired_per_fold.csv", index_label="fold")

    summary = {
        "folds": 7,
        "fresh_eeg_only_matches_archived_all_folds": True,
        "max_archive_difference": max_archive_difference,
        "arms": {},
    }
    for arm, frame in (("eeg_only", baseline), ("eye_clean_eeg_only", clean),
                       ("eeg_eye_fusion", fusion)):
        summary["arms"][arm] = {m: {"mean": float(frame[m].mean()),
                                     "sd_across_folds": float(frame[m].std())}
                                for m in METRICS}
        if arm != "eeg_only":
            summary["arms"][arm]["vs_eeg_only"] = {
                m: {"mean_delta": float((frame[m] - baseline[m]).mean()),
                    "positive_folds": int(((frame[m] - baseline[m]) > 0).sum())}
                for m in METRICS
            }
    (OUT / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
