from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def add_item_ids(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["item_id"] = (
        out["book"].astype(str)
        + "|run-"
        + out["run"].astype(int).astype(str).str.zfill(2)
        + "|row-"
        + out["row_idx"].astype(int).astype(str).str.zfill(4)
        + "|"
        + out["text"].astype(str)
    )
    return out


def assign_blocked_splits(meta: pd.DataFrame, fold: int = 0, n_blocks: int = 7) -> pd.DataFrame:
    """Assign item-disjoint train/val/test splits by contiguous text blocks."""
    out = add_item_ids(meta)
    item_rows = (
        out[["book", "run", "row_idx", "text", "item_id"]]
        .drop_duplicates("item_id")
        .sort_values(["book", "run", "row_idx", "text"])
        .reset_index(drop=True)
    )
    block_lookup: dict[str, int] = {}
    for _, group in item_rows.groupby("book", sort=False):
        indices = group.index.to_numpy()
        for block_id, block_indices in enumerate(np.array_split(indices, n_blocks)):
            for idx in block_indices:
                block_lookup[str(item_rows.loc[idx, "item_id"])] = block_id
    out["item_block"] = out["item_id"].map(block_lookup).astype(int)
    test_block = fold % n_blocks
    val_block = (fold + 1) % n_blocks
    out["split"] = "train"
    out.loc[out["item_block"] == val_block, "split"] = "val"
    out.loc[out["item_block"] == test_block, "split"] = "test"
    return out


def assign_book_run_group_splits(meta: pd.DataFrame, fold: int = 0, n_blocks: int = 7) -> pd.DataFrame:
    """Assign train/val/test by whole ``book+run`` groups.

    Cached metadata does not expose paragraph ids. This conservative split keeps
    every row from the same book/run group in exactly one split, preventing
    nearby reading samples from crossing train/validation/test while preserving
    item-disjointness.
    """
    out = add_item_ids(meta)
    out["book_run_group"] = out["book"].astype(str) + "|run-" + out["run"].astype(int).astype(str).str.zfill(2)
    groups = (
        out[["book", "run", "book_run_group"]]
        .drop_duplicates("book_run_group")
        .sort_values(["book", "run"])
        .reset_index(drop=True)
    )
    if groups.empty:
        raise RuntimeError("No book/run groups available for group-disjoint split.")
    if len(groups) < 3:
        raise RuntimeError("At least 3 book/run groups are required for train/val/test group-disjoint split.")
    n_blocks = min(max(3, int(n_blocks)), len(groups))
    group_to_block: dict[str, int] = {}
    for block_id, block_indices in enumerate(np.array_split(np.arange(len(groups)), n_blocks)):
        for idx in block_indices:
            group_to_block[str(groups.loc[int(idx), "book_run_group"])] = int(block_id)
    out["group_block"] = out["book_run_group"].map(group_to_block).astype(int)
    out["item_block"] = out["group_block"]
    test_block = fold % n_blocks
    val_block = (fold + 1) % n_blocks
    out["split"] = "train"
    out.loc[out["group_block"] == val_block, "split"] = "val"
    out.loc[out["group_block"] == test_block, "split"] = "test"
    return out


def assign_locked_book_run_group_splits(
    meta: pd.DataFrame,
    *,
    test_block: int,
    val_block: int,
    n_blocks: int = 7,
) -> pd.DataFrame:
    """Keep one book+run block locked as test while validation rotates."""
    out = assign_book_run_group_splits(meta, fold=test_block, n_blocks=n_blocks)
    effective_blocks = int(out["group_block"].nunique())
    test_block = int(test_block) % effective_blocks
    val_block = int(val_block) % effective_blocks
    if val_block == test_block:
        raise ValueError("Validation block must differ from the locked test block.")
    out["split"] = "train"
    out.loc[out["group_block"] == val_block, "split"] = "val"
    out.loc[out["group_block"] == test_block, "split"] = "test"
    return out


def assign_book_stratified_locked_group_splits(
    meta: pd.DataFrame,
    *,
    test_block: int,
    val_block: int,
    n_blocks: int = 7,
) -> pd.DataFrame:
    """Stratify non-test book/run groups while preserving the legacy test set.

    The locked test groups are first obtained from the established assignment
    and never changed. Remaining groups are allocated round-robin within each
    book using metadata only, reducing avoidable single-book validation blocks
    without reading targets or any signal features.
    """
    legacy = assign_locked_book_run_group_splits(
        meta,
        test_block=test_block,
        val_block=val_block,
        n_blocks=n_blocks,
    )
    effective_blocks = int(legacy["group_block"].nunique())
    test_block = int(test_block) % effective_blocks
    val_block = int(val_block) % effective_blocks
    if val_block == test_block:
        raise ValueError("Validation block must differ from the locked test block.")
    development_blocks = [block for block in range(effective_blocks) if block != test_block]
    locked_groups = set(
        legacy.loc[legacy["group_block"] == test_block, "book_run_group"].astype(str)
    )
    groups = (
        legacy[["book", "run", "book_run_group"]]
        .drop_duplicates("book_run_group")
        .sort_values(["book", "run"])
    )
    group_to_block = {group: test_block for group in locked_groups}
    for _, book_groups in groups.loc[
        ~groups["book_run_group"].astype(str).isin(locked_groups)
    ].groupby("book", sort=True):
        for position, group in enumerate(book_groups["book_run_group"].astype(str)):
            group_to_block[group] = development_blocks[position % len(development_blocks)]
    out = legacy.copy()
    out["group_block"] = out["book_run_group"].astype(str).map(group_to_block).astype(int)
    out["item_block"] = out["group_block"]
    out["split"] = "train"
    out.loc[out["group_block"] == val_block, "split"] = "val"
    out.loc[out["group_block"] == test_block, "split"] = "test"
    out["group_assignment"] = "book_stratified_locked_test_v2"
    return out


def assign_nested_development_splits(
    meta: pd.DataFrame,
    *,
    outer_val_block: int,
    inner_val_block: int,
    locked_test_block: int = 6,
    n_blocks: int = 7,
) -> pd.DataFrame:
    """Assign nested development splits while keeping block 6 inaccessible.

    Blocks other than the locked test are development data. The outer
    validation block estimates development-time generalization; an independent
    inner validation block is used for every selection decision.
    """
    if len({int(outer_val_block), int(inner_val_block), int(locked_test_block)}) != 3:
        raise ValueError("outer, inner and locked-test blocks must be distinct")
    base = assign_book_stratified_locked_group_splits(
        meta,
        test_block=locked_test_block,
        val_block=outer_val_block,
        n_blocks=n_blocks,
    )
    available = set(base["group_block"].astype(int).unique())
    requested = {int(outer_val_block), int(inner_val_block), int(locked_test_block)}
    if not requested.issubset(available):
        raise ValueError(f"Requested nested blocks are unavailable: {sorted(requested - available)}")
    out = base.copy()
    block = out["group_block"].astype(int)
    out["outer_split"] = "train"
    out.loc[block == int(outer_val_block), "outer_split"] = "val"
    out.loc[block == int(locked_test_block), "outer_split"] = "locked_test"
    out["inner_split"] = "train"
    out.loc[block == int(inner_val_block), "inner_split"] = "val"
    out.loc[block == int(outer_val_block), "inner_split"] = "outer_val"
    out.loc[block == int(locked_test_block), "inner_split"] = "locked_test"
    # Existing training code may consume split, but only the inner split is
    # permitted during selection. Outer validation is intentionally hidden.
    out["split"] = out["inner_split"]
    out["protocol_phase"] = "development_only_after_confirmatory_exposure"
    out["nested_outer_val_block"] = int(outer_val_block)
    out["nested_inner_val_block"] = int(inner_val_block)
    out["locked_test_block"] = int(locked_test_block)
    return out


def check_nested_development_split(meta: pd.DataFrame) -> None:
    required = {
        "book_run_group",
        "item_id",
        "outer_split",
        "inner_split",
        "protocol_phase",
    }
    missing = sorted(required - set(meta.columns))
    if missing:
        raise RuntimeError(f"Nested split is missing columns: {missing}")
    if set(meta["protocol_phase"].astype(str)) != {
        "development_only_after_confirmatory_exposure"
    }:
        raise RuntimeError("Nested development phase is not explicitly marked as exposed")
    for split_col, labels in (
        ("outer_split", ("train", "val", "locked_test")),
        ("inner_split", ("train", "val", "outer_val", "locked_test")),
    ):
        groups = {
            label: set(meta.loc[meta[split_col] == label, "book_run_group"].astype(str))
            for label in labels
        }
        items = {
            label: set(meta.loc[meta[split_col] == label, "item_id"].astype(str))
            for label in labels
        }
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                if groups[left] & groups[right]:
                    raise RuntimeError(f"Nested group leakage in {split_col}: {left}/{right}")
                if items[left] & items[right]:
                    raise RuntimeError(f"Nested item leakage in {split_col}: {left}/{right}")


def check_group_disjoint(meta: pd.DataFrame, group_col: str = "book_run_group") -> None:
    if group_col not in meta.columns:
        raise RuntimeError(f"Missing group column for group-disjoint audit: {group_col}")
    sets = {s: set(meta.loc[meta["split"] == s, group_col]) for s in ("train", "val", "test")}
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"Group leakage detected for {group_col}: {bad}")


def check_no_item_leakage(meta: pd.DataFrame) -> None:
    sets = {s: set(meta.loc[meta["split"] == s, "item_id"]) for s in ("train", "val", "test")}
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"Item leakage detected: {bad}")


def check_subject_disjoint(meta: pd.DataFrame) -> None:
    """Raise if any subject appears in both a train/val and the test split.

    This is the guard for the subject-disjoint (LOSO / dual-heldout) protocols:
    the test subject(s) must never contribute a single training or validation
    sample, otherwise per-subject neural signatures leak into model selection.
    """
    sets = {
        s: set(meta.loc[meta["split"] == s, "subject"])
        for s in ("train", "val", "test")
    }
    overlaps = {
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"Subject leakage detected (subject-disjoint protocol): {bad}")


def compute_split_audit(meta: pd.DataFrame) -> dict:
    """Compute a leakage/split audit summary for ``meta`` (Requirement 6.1).

    Returns a JSON-serializable dict with per-split unique ``item_id`` counts,
    per-split sample counts, pairwise train/val/test ``item_id`` overlap counts
    AND the analogous ``subject`` overlaps. Zero item overlaps confirm an
    item-disjoint split; zero subject overlaps confirm a subject-disjoint split.

    The ``protocol_kind`` / ``warnings`` fields make the honest interpretation
    explicit: a subject-disjoint-only split (LOSO) still shares items across
    train/test, so the per-item label structure can leak; the strict claim
    needs a dual-heldout (subject- AND item-disjoint) split.
    """
    splits = ("train", "val", "test")
    sets = {s: set(meta.loc[meta["split"] == s, "item_id"]) for s in splits}
    has_subject = "subject" in meta.columns
    subj = {
        s: set(meta.loc[meta["split"] == s, "subject"]) if has_subject else set()
        for s in splits
    }
    item_disjoint = bool(
        not (sets["train"] & sets["val"])
        and not (sets["train"] & sets["test"])
        and not (sets["val"] & sets["test"])
    )
    subject_disjoint = bool(
        has_subject
        and not (subj["train"] & subj["test"])
        and not (subj["val"] & subj["test"])
    )
    if subject_disjoint and item_disjoint:
        kind = "dual_heldout"
    elif subject_disjoint:
        kind = "subject_disjoint"
    elif item_disjoint:
        kind = "item_disjoint"
    else:
        kind = "non_disjoint"
    warnings: list[str] = []
    if kind == "subject_disjoint":
        warnings.append(
            "Subject-disjoint but items are SHARED across train/test: the "
            "per-item label (surprisal) structure can leak via memorization. "
            "Use the dual-heldout protocol for the strict headline claim."
        )
    if kind == "item_disjoint" and has_subject:
        warnings.append(
            "Item-disjoint but subjects are SHARED across train/test: "
            "per-subject neural signatures are seen at train time. Report a "
            "subject-disjoint protocol alongside this for credible generalization."
        )
    if kind == "non_disjoint":
        warnings.append("Neither item- nor subject-disjoint: results not credible.")
    audit = {
        "n_samples": int(len(meta)),
        "sample_counts_by_split": {
            s: int((meta["split"] == s).sum()) for s in splits
        },
        "unique_items_by_split": {s: int(len(sets[s])) for s in splits},
        "item_overlap_counts": {
            "train_val": int(len(sets["train"] & sets["val"])),
            "train_test": int(len(sets["train"] & sets["test"])),
            "val_test": int(len(sets["val"] & sets["test"])),
        },
        "item_disjoint": item_disjoint,
        "unique_subjects_by_split": {s: int(len(subj[s])) for s in splits},
        "subject_overlap_counts": {
            "train_test": int(len(subj["train"] & subj["test"])),
            "val_test": int(len(subj["val"] & subj["test"])),
        },
        "subject_disjoint": subject_disjoint,
        "protocol_kind": kind,
        "warnings": warnings,
    }
    if "book_run_group" in meta.columns:
        group_sets = {
            s: set(meta.loc[meta["split"] == s, "book_run_group"])
            for s in splits
        }
        group_overlaps = {
            "train_val": int(len(group_sets["train"] & group_sets["val"])),
            "train_test": int(len(group_sets["train"] & group_sets["test"])),
            "val_test": int(len(group_sets["val"] & group_sets["test"])),
        }
        audit["unique_book_run_groups_by_split"] = {
            s: int(len(group_sets[s]))
            for s in splits
        }
        audit["book_run_group_overlap_counts"] = group_overlaps
        audit["book_run_group_disjoint"] = bool(
            all(value == 0 for value in group_overlaps.values())
        )
    if "group_block" in meta.columns:
        audit["group_blocks_by_split"] = {
            s: sorted(int(value) for value in meta.loc[meta["split"] == s, "group_block"].unique())
            for s in splits
        }
    return audit


def write_split_audit(meta: pd.DataFrame, out_dir: "Path | str", run_name: str) -> "Path":
    """Write the split audit (see :func:`compute_split_audit`) as JSON.

    The file is saved to ``out_dir`` (typically ``results/logs/``) as
    ``{run_name}_split_audit.json`` and the path is returned.
    """
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = compute_split_audit(meta)
    audit["run_name"] = run_name
    path = out_dir / f"{run_name}_split_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return path


def assign_loso_splits(meta: pd.DataFrame, test_subject: str | None = None) -> pd.DataFrame:
    """Leave-one-subject-out split for subject generalization."""
    out = add_item_ids(meta)
    subjects = sorted(out["subject"].unique())
    if not subjects:
        raise RuntimeError("No subjects available for LOSO split.")
    test_subject = test_subject or subjects[-1]
    val_subject = next(s for s in subjects if s != test_subject)
    out["split"] = "train"
    out.loc[out["subject"] == val_subject, "split"] = "val"
    out.loc[out["subject"] == test_subject, "split"] = "test"
    out["item_block"] = -1
    return out


def assign_dual_heldout_splits(
    meta: pd.DataFrame,
    fold: int = 0,
    n_blocks: int = 7,
    test_subject: str | None = None,
) -> pd.DataFrame:
    """Hold out both a subject and a contiguous item block for strict testing."""
    out = assign_blocked_splits(meta, fold=fold, n_blocks=n_blocks)
    subjects = sorted(out["subject"].unique())
    test_subject = test_subject or subjects[-1]
    test_block = fold % n_blocks
    val_block = (fold + 1) % n_blocks
    out["split"] = "ignore"
    train_mask = (out["subject"] != test_subject) & (~out["item_block"].isin([test_block, val_block]))
    val_mask = (out["subject"] != test_subject) & (out["item_block"] == val_block)
    test_mask = (out["subject"] == test_subject) & (out["item_block"] == test_block)
    out.loc[train_mask, "split"] = "train"
    out.loc[val_mask, "split"] = "val"
    out.loc[test_mask, "split"] = "test"
    out = out[out["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    return out


def loso_test_subjects(meta: pd.DataFrame) -> list[str]:
    """Ordered list of subjects to leave out, one per LOSO-CV fold."""
    return sorted(str(s) for s in meta["subject"].unique())


def carve_calibration_split(
    meta: pd.DataFrame,
    n_calib_items: int = 30,
    seed: int = 2026,
) -> pd.DataFrame:
    """Carve a small, item-disjoint ``calib`` split out of the test subject's rows.

    For subject-adaptive (FiLM/adapter) models under LOSO/dual-heldout we need a
    handful of the held-out subject's trials to *estimate that subject's
    embedding* -- but those calibration trials must never be scored. This moves
    ``n_calib_items`` whole items (by ``item_id``) from ``split == "test"`` to a
    new ``split == "calib"`` value, guaranteeing the calibration items are
    disjoint from the remaining test items.

    The calibration trials are EEG/eye only by contract: downstream code must
    NOT read their ``surprisal``/``y_cogload`` labels (enforced in training,
    asserted by tests). The split is deterministic given ``seed``.
    """
    if "item_id" not in meta.columns:
        meta = add_item_ids(meta)
    out = meta.copy()
    test_mask = out["split"].to_numpy() == "test"
    test_items = sorted(out.loc[test_mask, "item_id"].unique())
    if len(test_items) <= n_calib_items:
        raise RuntimeError(
            f"Not enough test items ({len(test_items)}) to carve {n_calib_items} "
            "calibration items while leaving a non-empty test split."
        )
    rng = np.random.default_rng(seed)
    calib_items = set(rng.choice(test_items, size=n_calib_items, replace=False).tolist())
    calib_mask = test_mask & out["item_id"].isin(calib_items).to_numpy()
    out.loc[calib_mask, "split"] = "calib"
    return out


def limit_items_per_book(meta: pd.DataFrame, max_items_per_book: int | None) -> pd.DataFrame:
    if max_items_per_book is None:
        return meta
    keep: set[str] = set()
    tmp = add_item_ids(meta)
    item_rows = (
        tmp[["book", "run", "row_idx", "text", "item_id"]]
        .drop_duplicates("item_id")
        .sort_values(["book", "run", "row_idx", "text"])
    )
    for _, group in item_rows.groupby("book", sort=False):
        keep.update(group.head(max_items_per_book)["item_id"].tolist())
    return tmp[tmp["item_id"].isin(keep)].reset_index(drop=True)
