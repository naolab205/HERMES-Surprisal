#!/usr/bin/env python3
"""Summarize paired item metrics and a train-only length/layout oracle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cogalign.io.retrospective_splits import attach_retrospective_split  # noqa: E402
from foundation.data import CACHE_DIR, SURPRISAL_CSV, load_char_bags  # noqa: E402

OUT = ROOT / "results/response_mechanism_20261002"


def metrics(frame: pd.DataFrame) -> dict:
    if len(frame) < 3:
        raise ValueError("Too few observations for metrics")
    y = frame.y.to_numpy(dtype=float)
    p = frame.pred.to_numpy(dtype=float)
    return {"n": len(frame), "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(pearsonr(y, p).statistic),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


def item_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Preserve exact ties when identical predictions have different repeat counts.
    # Summing identical floats then dividing can otherwise invent tiny differences.
    def stable_mean(values):
        x = values.to_numpy(dtype=np.float64)
        return float(x[0] + np.mean(x - x[0]))
    return (frame.groupby("item_id", as_index=False)
            .agg(y=("y", "first"), pred=("pred", stable_mean),
                 book=("book", "first"), n_slots=("n_slots", "median")))


def conditional_rho(items: pd.DataFrame) -> float | None:
    groups = [(len(g), spearmanr(g.y, g.pred).statistic)
              for _, g in items.groupby(["book", "n_slots"])
              if len(g) >= 5 and np.ptp(g.pred.to_numpy()) >
              32 * np.finfo(float).eps * max(1., np.max(np.abs(g.pred)))
              and g.y.nunique() > 1]
    valid = [(n, r) for n, r in groups if np.isfinite(r)]
    if not valid:
        return None
    return float(np.average([r for _, r in valid],
                            weights=[n for n, _ in valid]))


def main() -> None:
    bags = load_char_bags(cache_dir=CACHE_DIR)
    manifest = pd.read_csv(OUT / "sentence_manifest.csv")
    cols = ["subject", "book", "run", "row_idx", "n_slots"]
    counts = bags.obs[cols]
    rows = []
    paired = []
    for fold in (0, 3):
        path = OUT / f"baseline_f{fold}/tables/charbag_retrospective_baseline_f{fold}_predictions.csv"
        prediction = pd.read_csv(path)
        if prediction.duplicated(cols[:4]).any():
            raise RuntimeError("Duplicate validation readings")
        pred = prediction.merge(counts, on=cols[:4], validate="one_to_one")
        source = attach_retrospective_split(bags.obs, manifest, fold,
                                            label_csv=str(SURPRISAL_CSV))
        train = source[source.split.eq("train")].copy()
        train["n_slots"] = train.n_slots.astype(int)
        category = train.groupby(["book", "n_slots"]).y.mean()
        fallback = train.groupby("book").y.mean()
        oracle = pred.copy()
        oracle["pred"] = [float(category.get((r.book, int(r.n_slots)),
                                             fallback[r.book]))
                          for r in pred.itertuples()]
        for name, frame in (("eeg_baseline", pred), ("train_length_oracle", oracle)):
            items = item_frame(frame)
            rows.append({"fold": fold, "arm": name,
                         "trial": metrics(frame), "item": metrics(items),
                         "book_count_spearman": conditional_rho(items)})
        eeg_items = item_frame(pred)
        oracle_items = item_frame(oracle)[["item_id", "pred"]].rename(
            columns={"pred": "oracle"})
        pair = eeg_items.merge(oracle_items, on="item_id", validate="one_to_one")
        pair = pair.merge(manifest[["item_id", "sentence"]], on="item_id",
                          validate="one_to_one")
        pair["sentence_group"] = pair.book + "|" + pair.sentence
        group_indices = {key: np.flatnonzero(pair.sentence_group.eq(key).to_numpy())
                         for key in pair.sentence_group.unique()}
        by_book = {book: pair.loc[pair.book.eq(book), "sentence_group"].unique()
                   for book in pair.book.unique()}
        rng = np.random.default_rng(2026 + fold)
        differences = []
        for _ in range(500):
            indices = np.concatenate([
                np.concatenate([group_indices[k] for k in rng.choice(keys, len(keys), replace=True)])
                for keys in by_book.values()
            ])
            sample = pair.iloc[indices]
            differences.append(float(spearmanr(sample.y, sample.pred).statistic -
                                     spearmanr(sample.y, sample.oracle).statistic))
        observed = float(spearmanr(pair.y, pair.pred).statistic -
                         spearmanr(pair.y, pair.oracle).statistic)
        paired.append({"fold": fold, "delta_item_spearman_eeg_minus_oracle": observed,
                       "sentence_cluster_bootstrap_95pct":
                           np.quantile(differences, [.025, .975]).tolist(),
                       "bootstrap_repetitions": 500})
    (OUT / "retrospective_baseline_summary.json").write_text(
        json.dumps({"scope": "folds 0 and 3, validation partitions only",
                    "rows": rows, "paired_item_analysis": paired}, indent=2) + "\n")
    print(json.dumps({"rows": rows, "paired_item_analysis": paired}, indent=2))


if __name__ == "__main__":
    main()
