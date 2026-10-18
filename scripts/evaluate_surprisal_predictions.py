#!/usr/bin/env python3
"""Audit held-out row-surprisal predictions beyond the original three metrics.

Scores are calculated separately by fold because each fold standardizes its
target with its own training-only mean and standard deviation. The script does
not treat folds or readers as independent significance-test replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)


def rank_correlation(values: pd.DataFrame) -> float:
    if values["y"].nunique() < 2 or values["pred"].nunique() < 2:
        return float("nan")
    return float(spearmanr(values["y"], values["pred"]).statistic)


def metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    y = frame["y"].to_numpy(dtype=float)
    pred = frame["pred"].to_numpy(dtype=float)
    if len(frame) < 3 or np.std(y) < 1e-9:
        raise ValueError(f"Insufficient or constant data for {prefix}")
    ccc = 2 * np.cov(y, pred, ddof=0)[0, 1] / (
        np.var(y) + np.var(pred) + (np.mean(y) - np.mean(pred)) ** 2
    )
    return {
        f"{prefix}_n": int(len(frame)),
        f"{prefix}_spearman": rank_correlation(frame),
        f"{prefix}_pearson": float(pearsonr(y, pred).statistic),
        f"{prefix}_kendall": float(kendalltau(y, pred).statistic),
        f"{prefix}_mae_train_sd": float(mean_absolute_error(y, pred)),
        f"{prefix}_rmse_train_sd": float(np.sqrt(mean_squared_error(y, pred))),
        f"{prefix}_r2": float(r2_score(y, pred)),
        f"{prefix}_ccc": float(ccc),
    }


def stratified_rank_rho(item: pd.DataFrame, columns: list[str]) -> float:
    # Within each nuisance stratum, map ranks to approximately Uniform(0,1)
    # and correlate only their within-stratum deviations. This removes
    # between-book and between-count shifts from the ranking endpoint.
    chunks = []
    for _, part in item.groupby(columns, sort=False):
        if len(part) < 5:
            continue
        ranks = part[["y", "pred"]].rank(method="average") / (len(part) + 1)
        chunks.append(ranks - .5)
    joined = pd.concat(chunks, ignore_index=True)
    return float(pearsonr(joined["y"], joined["pred"]).statistic)


def weighted_stratum_spearman(item: pd.DataFrame, columns: list[str]) -> float:
    scores = [(len(part), rank_correlation(part))
              for _, part in item.groupby(columns, sort=False) if len(part) >= 5]
    finite = [(n, rho) for n, rho in scores if np.isfinite(rho)]
    return float(np.average([rho for _, rho in finite],
                            weights=[n for n, _ in finite]))


def load_item_metadata(cache_dir: Path) -> pd.DataFrame:
    keys = ["book", "run", "row_idx"]
    frames = [pd.read_csv(path, usecols=keys + ["n_slots"])
              for path in sorted(cache_dir.glob("meta_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No character metadata under {cache_dir}")
    meta = pd.concat(frames, ignore_index=True)
    if meta.groupby(keys)["n_slots"].nunique().max() != 1:
        raise ValueError("Character count differs across readers for an item")
    return meta.drop_duplicates(keys)[keys + ["n_slots"]]


def evaluate(path: Path, out_dir: Path, cache_dir: Path | None = None) -> None:
    frame = pd.read_csv(path)
    required = {"subject", "book", "item_id", "y", "tercile", "pred", "fold"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if not np.isfinite(frame[["y", "pred"]].to_numpy()).all():
        raise ValueError("Non-finite target or prediction")
    if frame.groupby(["fold", "item_id"])["y"].nunique().max() != 1:
        raise ValueError("Target differs between readings of the same item")
    if frame.groupby(["fold", "item_id"])["tercile"].nunique().max() != 1:
        raise ValueError("Tercile differs between readings of the same item")
    if frame.groupby("item_id")["fold"].nunique().max() != 1:
        raise ValueError("An item appears in more than one held-out fold")

    item_meta = load_item_metadata(cache_dir) if cache_dir else None
    rows, book_rows, reader_rows = [], [], []
    for fold, trial in frame.groupby("fold", sort=True):
        item = trial.groupby("item_id", sort=False).agg(
            y=("y", "first"), pred=("pred", "mean"),
            tercile=("tercile", "first"), book=("book", "first"),
            run=("run", "first"), row_idx=("row_idx", "first"),
            n_readers=("pred", "size"),
        ).reset_index()
        if item_meta is not None:
            item = item.merge(item_meta, on=["book", "run", "row_idx"],
                              how="left", validate="one_to_one")
            if item["n_slots"].isna().any():
                raise ValueError("Character count metadata missing for evaluated item")
        record = {"fold": int(fold), **metrics(trial, "trial"), **metrics(item, "item")}
        if item_meta is not None:
            stratum_sizes = item.groupby(["book", "n_slots"]).size()
            record["item_book_count_covered"] = int(stratum_sizes[stratum_sizes >= 5].sum())
            record["item_book_count_strata"] = int((stratum_sizes >= 5).sum())
            record["item_within_book_rank_rho"] = stratified_rank_rho(item, ["book"])
            record["item_within_book_count_rank_rho"] = stratified_rank_rho(
                item, ["book", "n_slots"]
            )
            record["item_weighted_book_count_spearman"] = weighted_stratum_spearman(
                item, ["book", "n_slots"]
            )
            record["item_count_oracle_spearman"] = float(
                spearmanr(item["y"], item["n_slots"]).statistic
            )
        extreme = item[item["tercile"] != 1]
        binary = (extreme["tercile"] == 2).astype(int)
        if binary.nunique() != 2:
            raise ValueError(f"Only one extreme tercile in fold {fold}")
        record["item_auc_hi_vs_lo"] = float(roc_auc_score(binary, extreme["pred"]))
        record["item_average_precision_hi_vs_lo"] = float(
            average_precision_score(binary, extreme["pred"])
        )
        record["item_median_readings"] = float(item["n_readers"].median())
        rows.append(record)
        for book, subset in item.groupby("book"):
            book_rows.append({"fold": int(fold), "book": book,
                              **metrics(subset, "item")})
        for subject, subset in trial.groupby("subject"):
            reader_rows.append({"fold": int(fold), "subject": subject,
                                **metrics(subset, "trial")})

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(out_dir / "extra_metrics_per_fold.csv", index=False)
    pd.DataFrame(book_rows).to_csv(out_dir / "extra_metrics_by_book.csv", index=False)
    pd.DataFrame(reader_rows).to_csv(out_dir / "extra_metrics_by_reader.csv", index=False)
    summary = {"source": str(path), "n_folds": len(fold_frame),
               "n_readings": int(len(frame)),
               "n_unique_items": int(frame["item_id"].nunique()),
               "metrics": {c: {"mean": float(fold_frame[c].mean()),
                               "sd_across_folds": float(fold_frame[c].std())}
                           for c in fold_frame.columns if c != "fold"}}
    (out_dir / "extra_metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_folds": summary["n_folds"],
                      "n_unique_items": summary["n_unique_items"],
                      "item_metrics": {k: v for k, v in summary["metrics"].items()
                                       if k.startswith("item_")}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        help="character metadata for within-book/count controls")
    args = parser.parse_args()
    evaluate(args.predictions, args.out_dir, args.cache_dir)
