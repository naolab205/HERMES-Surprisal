#!/usr/bin/env python3
"""Paired split-reader reliability on saved deconvolution validation outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_schedule_deconv import BASE, read_arm  # noqa: E402
from evaluate_surprisal_predictions import load_item_metadata  # noqa: E402


def conditional_half_correlation(a: list[float], b: list[float],
                                 strata: pd.DataFrame) -> float:
    table = strata.copy()
    table["a"], table["b"] = a, b
    parts = []
    for _, group in table.groupby(["book", "n_slots"]):
        if len(group) < 5:
            continue
        ranks = group[["a", "b"]].rank() / (len(group) + 1) - .5
        parts.append(ranks)
    joined = pd.concat(parts, ignore_index=True)
    return float(np.corrcoef(joined.a, joined.b)[0, 1])


def main() -> None:
    arms = {name: read_arm(name) for name in ("control", "candidate")}
    item_meta = load_item_metadata(Path("/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms"))
    rows = []
    for fold in range(7):
        mats = {name: frame[frame.fold == fold].pivot(index="item_id", columns="subject", values="pred")
                for name, frame in arms.items()}
        if not mats["control"].index.equals(mats["candidate"].index):
            raise ValueError("Item mismatch")
        if not mats["control"].columns.equals(mats["candidate"].columns):
            raise ValueError("Reader mismatch")
        x = {name: matrix.to_numpy(dtype=float) for name, matrix in mats.items()}
        strata = (arms["control"].loc[arms["control"].fold == fold,
                                      ["item_id", "book", "run", "row_idx"]]
                  .drop_duplicates("item_id")
                  .merge(item_meta, on=["book", "run", "row_idx"], validate="one_to_one")
                  .set_index("item_id").loc[mats["control"].index, ["book", "n_slots"]]
                  .reset_index(drop=True))
        rng = np.random.default_rng(20261002 + fold)
        reliabilities = {name: [] for name in arms}
        conditional = {name: [] for name in arms}
        for _ in range(30):
            halves = {name: ([], []) for name in arms}
            for i in range(len(mats["control"])):
                present = np.flatnonzero(np.isfinite(x["control"][i]))
                if len(present) < 4:
                    continue
                take = rng.permutation(present)[:2 * (len(present) // 2)]
                a, b = np.array_split(take, 2)
                for name in arms:
                    halves[name][0].append(x[name][i, a].mean())
                    halves[name][1].append(x[name][i, b].mean())
            for name, (a, b) in halves.items():
                r = float(spearmanr(a, b).statistic)
                reliabilities[name].append(2 * r / (1 + r))
                r_cond = conditional_half_correlation(a, b, strata)
                conditional[name].append(2 * r_cond / (1 + r_cond))
        rows.append({"fold": fold,
                     "control_split_half": float(np.mean(reliabilities["control"])),
                     "candidate_split_half": float(np.mean(reliabilities["candidate"])),
                     "delta_split_half": float(np.mean(reliabilities["candidate"])
                                               - np.mean(reliabilities["control"])),
                     "control_conditional_split_half": float(np.mean(conditional["control"])),
                     "candidate_conditional_split_half": float(np.mean(conditional["candidate"])),
                     "delta_conditional_split_half": float(np.mean(conditional["candidate"])
                                                           - np.mean(conditional["control"])),
                     "control_mean_within_item_sd": float(np.nanmean(np.nanstd(x["control"], axis=1))),
                     "candidate_mean_within_item_sd": float(np.nanmean(np.nanstd(x["candidate"], axis=1)))})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "reader_consistency.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("means", out.mean(numeric_only=True).round(4).to_dict())


if __name__ == "__main__":
    main()
