from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import BOOK_TO_SESSION


def pair_rows_rowe(events_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each ROWS marker with the next ROWE marker.

    The row index is the ordinal position among post-preface ROWS events. Rows
    with implausible duration are dropped without shifting later row indices.
    """
    rows = (
        events_df[events_df["trial_type"] == "ROWS"]
        .sort_values("onset")
        .reset_index(drop=True)
    )
    rowe = (
        events_df[events_df["trial_type"] == "ROWE"]
        .sort_values("onset")
        .reset_index(drop=True)
    )
    pree = events_df[events_df["trial_type"] == "PREE"]
    ch_events = events_df[events_df["trial_type"].astype(str).str.startswith("CH")]
    if len(pree) > 0:
        cutoff = float(pree.iloc[0]["onset"])
        rows = rows[rows["onset"] > cutoff].reset_index(drop=True)
        rowe = rowe[rowe["onset"] > cutoff].reset_index(drop=True)
    elif len(ch_events) > 0:
        cutoff = float(ch_events.iloc[0]["onset"])
        rows = rows[rows["onset"] >= cutoff - 1.0].reset_index(drop=True)

    pairs: list[dict[str, float | int]] = []
    for i in range(len(rows)):
        start = float(rows.iloc[i]["onset"])
        after = rowe[rowe["onset"] > start]
        if len(after) == 0:
            continue
        end = float(after.iloc[0]["onset"])
        dur = end - start
        if 1.0 < dur < 6.0:
            pairs.append({"row_idx": i, "start": start, "end": end, "dur": dur})
    return pd.DataFrame(pairs)


def raw_events_path(data_root: Path, subject: str, book: str, run: int) -> Path:
    session = BOOK_TO_SESSION[book]
    return (
        data_root
        / subject
        / session
        / "eeg"
        / f"{subject}_{session}_task-reading_run-{run:02d}_events.tsv"
    )


def preproc_events_path(data_root: Path, subject: str, book: str, run: int) -> Path:
    session = BOOK_TO_SESSION[book]
    return (
        data_root
        / "derivatives"
        / "preproc"
        / "filtered_0.5_30"
        / subject
        / session
        / "eeg"
        / f"{subject}_{session}_task-reading_run-{run:02d}_events.tsv"
    )


def text_embedding_path(data_root: Path, book: str, run: int) -> Path:
    return (
        data_root
        / "derivatives"
        / "text_embeddings"
        / f"{book}_text_embedding"
        / f"text_embedding_run_{run}.npy"
    )


def safe_read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["onset", "duration", "trial_type", "value", "sample"])
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def load_text_embedding(data_root: Path, book: str, run: int) -> np.ndarray | None:
    path = text_embedding_path(data_root, book, run)
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)
