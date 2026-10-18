#!/usr/bin/env python3
"""Audit paired seven-fold validation predictions for scheduled deconvolution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from evaluate_surprisal_predictions import load_item_metadata, weighted_stratum_spearman

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/schedule_deconv_validation_20261002"
CACHE = Path("/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms")


def read_arm(arm: str) -> pd.DataFrame:
    paths = [("control" if arm == "control" else "candidate", f"deconv_{arm}_f0"),
             (f"{arm}_f3", f"deconv_{arm}_f3"),
             (f"{arm}_remaining", f"deconv_{arm}_remaining")]
    frames = [pd.read_csv(BASE / directory / "tables" / f"charbag_{name}_predictions.csv")
              for directory, name in paths]
    frame = pd.concat(frames, ignore_index=True)
    if sorted(frame.fold.unique()) != list(range(7)) or set(frame.eval_split) != {"val"}:
        raise ValueError(f"Incomplete or non-validation folds for {arm}")
    keys = ["fold", "subject", "book", "run", "row_idx"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate reading in {arm}")
    return frame.sort_values(keys).reset_index(drop=True)


def score(trials: pd.DataFrame) -> dict[str, float]:
    item = trials.groupby("item_id").agg(y=("y", "first"), pred=("pred", "mean"),
                                          tercile=("tercile", "first"),
                                          book=("book", "first"),
                                          n_slots=("n_slots", "first"))
    extreme = item[item.tercile != 1]
    return {"readings": int(len(trials)), "items": int(len(item)),
            "trial_rho": float(spearmanr(trials.y, trials.pred).statistic),
            "item_rho": float(spearmanr(item.y, item.pred).statistic),
            "book_count_rho": weighted_stratum_spearman(
                item.reset_index(), ["book", "n_slots"]),
            "item_auc": float(roc_auc_score((extreme.tercile == 2), extreme.pred))}


def main() -> None:
    control, candidate = read_arm("control"), read_arm("candidate")
    keys = ["fold", "subject", "book", "run", "row_idx", "item_id", "y", "tercile"]
    if not control[keys].equals(candidate[keys]):
        raise ValueError("Control and candidate evaluation rows/labels differ")
    meta = load_item_metadata(CACHE)
    control = control.merge(meta, on=["book", "run", "row_idx"], validate="many_to_one")
    candidate = candidate.merge(meta, on=["book", "run", "row_idx"], validate="many_to_one")
    rows, books = [], []
    for fold in range(7):
        a = control[control.fold == fold]
        b = candidate[candidate.fold == fold]
        x, y = score(a), score(b)
        rows.append({"fold": fold, **{f"control_{k}": v for k, v in x.items()},
                     **{f"candidate_{k}": v for k, v in y.items()},
                     **{f"delta_{k}": y[k] - x[k] for k in
                        ("trial_rho", "item_rho", "book_count_rho", "item_auc")}})
        for book in sorted(a.book.unique()):
            x, y = score(a[a.book == book]), score(b[b.book == book])
            books.append({"fold": fold, "book": book,
                          "control_item_rho": x["item_rho"],
                          "candidate_item_rho": y["item_rho"],
                          "delta_item_rho": y["item_rho"] - x["item_rho"],
                          "control_count_rho": weighted_stratum_spearman(
                              a[a.book == book].groupby("item_id").agg(
                                  y=("y", "first"), pred=("pred", "mean"),
                                  n_slots=("n_slots", "first")).reset_index(), ["n_slots"]),
                          "candidate_count_rho": weighted_stratum_spearman(
                              b[b.book == book].groupby("item_id").agg(
                                  y=("y", "first"), pred=("pred", "mean"),
                                  n_slots=("n_slots", "first")).reset_index(), ["n_slots"])})
    per_fold = pd.DataFrame(rows)
    by_book = pd.DataFrame(books)
    by_book["delta_count_rho"] = by_book.candidate_count_rho - by_book.control_count_rho
    per_fold.to_csv(BASE / "paired_validation_folds.csv", index=False)
    by_book.to_csv(BASE / "paired_validation_books.csv", index=False)
    means = {k: float(per_fold[k].mean()) for k in per_fold if k.startswith("delta_")}
    positive = {k: int((per_fold[k] > 0).sum()) for k in per_fold if k.startswith("delta_")}
    gate = (positive["delta_item_rho"] >= 5
            and positive["delta_book_count_rho"] >= 5
            and means["delta_item_rho"] >= .01
            and means["delta_book_count_rho"] >= .01
            and means["delta_trial_rho"] >= 0)
    report = {"n_folds": 7, "n_eval_readings": int(len(control)),
              "n_eval_items": int(control.item_id.nunique()),
              "mean_paired_differences": means, "positive_folds": positive,
              "predeclared_exploratory_gate_passed": bool(gate),
              "scope": "seven development validation blocks; not a sealed outer test"}
    (BASE / "paired_validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
