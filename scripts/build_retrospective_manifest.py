#!/usr/bin/env python3
"""Freeze and audit sentence-disjoint, purged development partitions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cogalign.io.retrospective_splits import (  # noqa: E402
    audit_cross_block_near_duplicates, sentence_manifest,
)
from foundation.data import CACHE_DIR, load_char_bags  # noqa: E402


def main() -> None:
    bags = load_char_bags(cache_dir=CACHE_DIR)
    manifest = sentence_manifest(bags.obs)
    out = ROOT / "results/response_mechanism_20261002"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sentence_manifest.csv"
    manifest.to_csv(path, index=False)
    included = manifest[manifest.included]
    near_duplicates = audit_cross_block_near_duplicates(manifest)
    if near_duplicates:
        raise RuntimeError(f"Near-duplicate sentences cross blocks: {near_duplicates[:3]}")
    # One frozen-LM token per Chinese character in this cache. Check the
    # *measured* text gap rather than assuming 24 rows always exceed 192 tokens.
    text = bags.obs.drop_duplicates(["book", "run", "row_idx"])[
        ["book", "run", "row_idx", "text"]]
    with_text = manifest.merge(text, on=["book", "run", "row_idx"],
                               validate="one_to_one")
    gap_chars = []
    for _, group in with_text.groupby(["book", "run"]):
        kept = group[group.included]
        left = kept.loc[kept.half.eq(0), "row_idx"].max()
        right = kept.loc[kept.half.eq(1), "row_idx"].min()
        gap = group[group.row_idx.gt(left) & group.row_idx.lt(right)]
        gap_chars.append(int(gap.text.str.len().sum()))
    if min(gap_chars) < 192:
        raise RuntimeError("At least one split boundary has <192 character tokens")
    audit = {
        "status": "retrospective development; no untouched outer data",
        "protocol": "two sentence-bound chunks per book/run, seven book-stratified blocks",
        "context_purge_rows_each_side": 24,
        "min_boundary_gap_character_tokens": min(gap_chars),
        "n_items_total": len(manifest),
        "n_items_included": len(included),
        "excluded_by_reason": manifest.loc[~manifest.included, "excluded_reason"].value_counts().to_dict(),
        "items_by_book_block": {f"{book}|{block}": int(n)
                                for (book, block), n in included.groupby(["book", "block"]).size().items()},
        "cross_block_sentence_overlap": int((included.groupby(["book", "sentence"]).block.nunique() > 1).sum()),
        "cross_block_near_duplicate_families": len(near_duplicates),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert audit["cross_block_sentence_overlap"] == 0
    (out / "manifest_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
