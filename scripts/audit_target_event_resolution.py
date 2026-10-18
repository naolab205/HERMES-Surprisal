#!/usr/bin/env python3
"""Check whether row labels and physical character events support this target."""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foundation.data import CACHE_DIR, SURPRISAL_CSV, load_char_bags  # noqa: E402


def expected_slots(text: str) -> int:
    return sum(not unicodedata.category(ch).startswith("P") and not ch.isspace()
               for ch in text)


def main() -> None:
    bags = load_char_bags(cache_dir=CACHE_DIR)
    labels = pd.read_csv(SURPRISAL_CSV,
                         usecols=["book", "run", "row_idx", "surprisal"])
    obs = bags.obs.merge(labels, on=["book", "run", "row_idx"],
                         validate="many_to_one")
    obs["expected_slots"] = obs.text.map(expected_slots)
    obs["missing_slots"] = obs.expected_slots - obs.n_slots
    if (obs.missing_slots < 0).any():
        raise RuntimeError("Cache has more character events than display slots")
    obs["punctuation_count"] = obs.text.str.len() - obs.expected_slots
    items = (obs.groupby(["book", "run", "row_idx", "text"], as_index=False)
             .agg(y=("surprisal", "first"), n_slots=("n_slots", "median"),
                  expected_slots=("expected_slots", "first"),
                  n_char=("n_char", "first"),
                  punctuation=("punctuation_count", "first"),
                  missing_readings=("missing_slots", lambda x: int((x > 0).sum())),
                  n_readers=("subject", "nunique"),
                  duration=("dur", "median")))
    # SOA is metadata only, never a predictor. It tests onset spread and
    # whether the fixed-window ERP pooling assumption is plausible.
    items["soa"] = items.duration / items.expected_slots
    correlations = {}
    for field in ("n_slots", "expected_slots", "n_char", "punctuation", "soa"):
        correlations[field] = float(spearmanr(items.y, items[field]).statistic)
    result = {
        "n_items": len(items), "n_readings": len(obs),
        "row_label_granularity": "mean token negative log likelihood including punctuation",
        "eeg_event_granularity": "one epoch per highlighted nonpunctuation character",
        "readings_with_missing_display_slots": int((obs.missing_slots > 0).sum()),
        "fraction_readings_with_missing_slots": float((obs.missing_slots > 0).mean()),
        "punctuation_share_of_target_tokens": float(items.punctuation.sum() /
                                                      items.n_char.sum()),
        "items_with_punctuation": int((items.punctuation > 0).sum()),
        "soa_seconds_quantiles": {str(q): float(items.soa.quantile(q))
                                 for q in (.01, .25, .5, .75, .99)},
        "surprisal_spearman_with_metadata_audit_only": correlations,
        "by_book": {book: {"n_items": len(group),
                           "y_mean": float(group.y.mean()),
                           "y_std": float(group.y.std()),
                           "soa_median": float(group.soa.median())}
                    for book, group in items.groupby("book")},
    }
    out = ROOT / "results/response_mechanism_20261002/target_event_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
