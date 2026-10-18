#!/usr/bin/env python3
"""Audit character labels against the frozen row target before local supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-labels", type=Path,
                    default=Path("/data/2_readEEG/superisal_code/data_cache/char_surprisal.csv"))
    ap.add_argument("--row-labels", type=Path,
                    default=ROOT / "data_cache/surprisal_1829a99ad4.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/exploration_20261003/character_identity_audit.json")
    args = ap.parse_args()
    table = pd.read_csv(args.char_labels)
    rows = pd.read_csv(args.row_labels)
    grouped = (table.groupby(["book", "run", "row_idx"])
               .agg(total=("char_surprisal", "sum"), n_slots=("char_slot", "size"),
                    n_tokens=("n_tokens", "first"),
                    token_count_variants=("n_tokens", "nunique")).reset_index())
    if grouped.token_count_variants.ne(1).any():
        raise ValueError("Inconsistent token count within a row")
    checked = grouped.merge(rows[["book", "run", "row_idx", "surprisal"]],
                            on=["book", "run", "row_idx"], validate="one_to_one")
    if len(checked) != len(grouped):
        raise ValueError("A cached character-label item has no frozen row label")
    old_error = np.abs(checked.total / checked.n_slots - checked.surprisal)
    correct_error = np.abs(checked.total / checked.n_tokens - checked.surprisal)
    if float(correct_error.max()) > .01:
        raise ValueError("Stored character labels disagree with frozen row labels by >0.01 nats")
    result = {
        "n_rows": len(checked),
        "n_frozen_labels_outside_character_cache": len(rows) - len(checked),
        "n_rows_different_denominator": int(checked.n_slots.ne(checked.n_tokens).sum()),
        "slot_mean_mae_nats": float(old_error.mean()),
        "slot_mean_max_error_nats": float(old_error.max()),
        "token_mean_mae_nats": float(correct_error.mean()),
        "token_mean_max_error_nats": float(correct_error.max()),
        "char_labels_sha256": hashlib.sha256(args.char_labels.read_bytes()).hexdigest(),
        "row_labels_sha256": hashlib.sha256(args.row_labels.read_bytes()).hexdigest(),
        "conclusion": "Existing load_char_targets slot-mean identity is inconsistent; "
                      "repair contribution units before new character-supervised experiments",
        "performance_claim": "none; algebraic label audit only",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
