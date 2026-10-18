#!/usr/bin/env python3
"""Paired audit and preregistered gate for the local-supervision screen."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foundation.data import CACHE_DIR, load_char_bags  # noqa: E402
from summarize_retrospective_baseline import conditional_rho, item_frame, metrics  # noqa: E402

BASE = ROOT / "results/exploration_20261003/local_supervision"
KEYS = ["fold", "subject", "book", "run", "row_idx", "item_id", "y", "tercile"]


def score(frame: pd.DataFrame) -> dict:
    items = item_frame(frame)
    return {"trial": metrics(frame), "item": metrics(items),
            "book_count_rho": conditional_rho(items)}


def main() -> None:
    counts = load_char_bags(cache_dir=CACHE_DIR).obs[
        ["subject", "book", "run", "row_idx", "n_slots"]]
    arms = {}
    for arm in ("local_row", "local_real", "local_shuffle"):
        frame = pd.read_csv(BASE / arm / "tables" / f"charbag_{arm}_predictions.csv")
        if sorted(frame.fold.unique()) != [0, 3] or set(frame.eval_split) != {"val"}:
            raise RuntimeError("Incomplete/nonvalidation screen predictions")
        if frame.duplicated(KEYS[:5]).any() or not np.isfinite(frame[["y", "pred"]]).all().all():
            raise RuntimeError("Duplicate/nonfinite screen predictions")
        arms[arm] = frame.sort_values(KEYS[:5]).reset_index(drop=True)
    reference = pd.concat([
        pd.read_csv(ROOT / f"results/response_mechanism_20261002/baseline_f{fold}/tables/"
                    f"charbag_retrospective_baseline_f{fold}_predictions.csv")
        for fold in (0, 3)
    ]).sort_values(KEYS[:5]).reset_index(drop=True)
    arms["reference_row_only"] = reference
    for arm, frame in arms.items():
        if not frame[KEYS].equals(arms["local_row"][KEYS]):
            raise RuntimeError(f"Evaluation rows or frozen targets differ in {arm}")
        arms[arm] = frame.merge(counts, on=KEYS[1:5], validate="many_to_one")
    scores, differences = [], []
    for fold in (0, 3):
        by_arm = {arm: score(frame[frame.fold.eq(fold)]) for arm, frame in arms.items()}
        scores.extend({"fold": fold, "arm": arm, **value} for arm, value in by_arm.items())
        for control in ("local_row", "local_shuffle", "reference_row_only"):
            a, b = by_arm[control], by_arm["local_real"]
            differences.append({"fold": fold, "control": control,
                                "delta_trial_rho": b["trial"]["spearman"] - a["trial"]["spearman"],
                                "delta_item_rho": b["item"]["spearman"] - a["item"]["spearman"],
                                "delta_book_count_rho": b["book_count_rho"] - a["book_count_rho"]})
    gate = all(r["delta_item_rho"] > 0 and r["delta_book_count_rho"] > 0 and
               (r["control"] == "local_shuffle" or r["delta_trial_rho"] >= 0)
               for r in differences)
    report = {"scope": "fixed development folds 0/3, seed 2026; retrospective",
              "scores": scores, "paired_differences": differences,
              "localization_gate_passed": gate}
    pd.DataFrame(differences).to_csv(BASE / "paired_differences.csv", index=False)
    (BASE / "screen_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"paired_differences": differences,
                      "localization_gate_passed": gate}, indent=2))


if __name__ == "__main__":
    main()
