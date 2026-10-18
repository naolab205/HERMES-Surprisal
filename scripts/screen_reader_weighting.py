#!/usr/bin/env python3
"""Cross-validation-block screen of reader-quality item aggregation.

Reader weights are learned on the *other* validation block and applied without
retuning. This explores whether simple quality weighting can help an existing
decoder; it is neither a neural architecture experiment nor a test result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluate_surprisal_predictions import (load_item_metadata,
                                             weighted_stratum_spearman)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "results/overlap_demix_validation_20260930/tables/"
          "charbag_baseline_val_predictions.csv")
OUT = ROOT / "results/cross_reader_probe_20261002/reader_weighting.json"


def score(trials: pd.DataFrame, weight: dict[str, float] | None = None) -> dict:
    if weight is None:
        values = trials.groupby("item_id").agg(y=("y", "first"), pred=("pred", "mean"),
                                               book=("book", "first"),
                                               n_slots=("n_slots", "first"))
    else:
        copy = trials.copy()
        copy["weighted"] = copy.pred * copy.subject.map(weight)
        copy["weight"] = copy.subject.map(weight)
        values = copy.groupby("item_id").agg(y=("y", "first"),
                                             weighted=("weighted", "sum"),
                                             weight=("weight", "sum"),
                                             book=("book", "first"),
                                             n_slots=("n_slots", "first"))
        values["pred"] = values.weighted / values.weight
    return {"item_rho": float(spearmanr(values.y, values.pred).statistic),
            "book_count_rho": weighted_stratum_spearman(
                values.reset_index(), ["book", "n_slots"])}


def main() -> None:
    all_rows = pd.read_csv(SOURCE)
    meta = load_item_metadata(Path("/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms"))
    all_rows = all_rows.merge(meta, on=["book", "run", "row_idx"], validate="many_to_one")
    folds = sorted(all_rows.fold.unique())
    if folds != [0, 3]:
        raise ValueError(f"Expected folds 0 and 3, got {folds}")
    result = {"source": str(SOURCE), "rule": "nonnegative other-fold reader within-book/count rho, no tuning",
              "folds": {}}
    for dest in folds:
        source = all_rows[all_rows.fold != dest]
        target = all_rows[all_rows.fold == dest]
        weights = {subject: max(0.001, weighted_stratum_spearman(
            group, ["book", "n_slots"]))
                   for subject, group in source.groupby("subject")}
        result["folds"][str(dest)] = {"source_fold": int(source.fold.iloc[0]),
                                      "weights": weights,
                                      "equal": score(target),
                                      "reader_weighted": score(target, weights)}
    for endpoint in ("item_rho", "book_count_rho"):
        result[f"mean_delta_{endpoint}"] = float(np.mean([
            part["reader_weighted"][endpoint] - part["equal"][endpoint]
            for part in result["folds"].values()]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
