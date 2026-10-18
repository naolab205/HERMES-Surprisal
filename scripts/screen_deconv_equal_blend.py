#!/usr/bin/env python3
"""Post hoc, fixed 50/50 prediction blend of paired validation models.

This is an exploratory reconciliation screen after the deconvolution gate
failed. It reads no test rows, fits no weights, and must not be treated as an
independent validation of a newly chosen ensemble.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_surprisal_predictions import load_item_metadata
from summarize_schedule_deconv import BASE, CACHE, read_arm, score


def main() -> None:
    a, b = read_arm("control"), read_arm("candidate")
    keys = ["fold", "subject", "book", "run", "row_idx", "item_id", "y", "tercile"]
    if not a[keys].equals(b[keys]):
        raise ValueError("Paired prediction keys do not match")
    blend = a.copy()
    blend["pred"] = (a.pred.to_numpy() + b.pred.to_numpy()) / 2
    meta = load_item_metadata(CACHE)
    arms = {"control": a, "deconvolution": b, "equal_blend": blend}
    arms = {name: frame.merge(meta, on=["book", "run", "row_idx"], validate="many_to_one")
            for name, frame in arms.items()}
    rows, book_rows = [], []
    for fold in range(7):
        result = {name: score(frame[frame.fold == fold]) for name, frame in arms.items()}
        rows.append({"fold": fold, **{f"{name}_{metric}": value
                                      for name, values in result.items()
                                      for metric, value in values.items()
                                      if metric in ("trial_rho", "item_rho", "book_count_rho", "item_auc")}})
        for book in sorted(blend.book.unique()):
            result = {name: score(frame[(frame.fold == fold) & (frame.book == book)])
                      for name, frame in arms.items()}
            book_rows.append({"fold": fold, "book": book,
                              **{f"{name}_{metric}": value
                                 for name, values in result.items()
                                 for metric, value in values.items()
                                 if metric in ("item_rho", "book_count_rho")}})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(BASE / "equal_blend_validation_folds.csv", index=False)
    pd.DataFrame(book_rows).to_csv(BASE / "equal_blend_validation_books.csv", index=False)
    summary = {name: {metric: float(per_fold[f"{name}_{metric}"].mean())
                      for metric in ("trial_rho", "item_rho", "book_count_rho", "item_auc")}
               for name in arms}
    summary["scope"] = "post hoc equal blend on seven development validation blocks"
    (BASE / "equal_blend_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
