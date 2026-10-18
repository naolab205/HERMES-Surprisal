#!/usr/bin/env python
"""Build a character-locked EEG epoch cache from the ChineseEEG preproc derivative.

Why this exists
---------------
The frozen JBHI pipeline resampled every reading row to a fixed 128 samples
regardless of its 1-6 s duration (``build_epoch_cache.py``).  That destroys the
physical time axis: the effective sampling rate becomes ``128 / dur`` and varies
per trial, ERP latencies are time-warped, and band-power features computed at an
assumed 128 Hz describe a different frequency range on every row.

The ChineseEEG paradigm is program-paced: each non-punctuation character of the
highlighted row is coloured red for a fixed interval (dataset README).  The
measured stimulus-onset asynchrony is ~0.42 s and the row duration is a near
deterministic function of the highlighted character count (r = 0.986).  Character
onsets are therefore recoverable without eye tracking, and a character-locked
average shows a textbook visual evoked response (P1 ~95 ms, N1 ~150 ms, and the
next character's P1 at SOA + 95 ms).

This builder emits one epoch per highlighted character on the *native* physical
time axis.  The row becomes a variable-length bag of fixed-length character
epochs: no time warping and no duration-dependent bandwidth. Training still
uses one row-level target per reading; the epochs add no independent labels.

Source data
-----------
``derivatives/preproc/filtered_0.5_30`` is the dataset's fully pre-processed
tier: 256 Hz, 0.5-30 Hz band-pass, bad-channel interpolation, ICA artifact
removal and average reference already applied by the dataset authors.

Leakage contract
----------------
The stored array contains signal only.  ``dur``, ``n_char``, ``soa``,
``row_onset`` and ``char_idx`` are written to the metadata table for auditing
and aggregation, and must never be fed to the neural encoder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
REF_SRC = ROOT / "src"
sys.path.insert(0, str(REF_SRC))

sys.path.insert(0, str(REF_SRC.parent))

from cogalign.core.config import BOOK_TO_SESSION  # noqa: E402
from cogalign.io.events import pair_rows_rowe, preproc_events_path  # noqa: E402
from scripts.build.build_epoch_cache import _read_raw_brainvision_robust  # noqa: E402

DATA_ROOT = Path(os.environ.get("CHINESEEEG_ROOT", "/data/my/ds004952"))
SOURCE_SFREQ = 256.0
TARGET_SFREQ = 128.0  # data is band-limited to 30 Hz; 128 Hz is 4x Nyquist
PRE_SECONDS = 0.100
POST_SECONDS = 0.450
AMPLITUDE_CLIP_UV = 300.0
N_CHANNELS = 128


def highlight_indices(text: str) -> list[int]:
    """Positions of the characters that receive their own red highlight.

    The paradigm highlights each Chinese character of the row in turn.  Marks
    and whitespace ride along with the preceding character rather than getting
    their own presentation slot, which is why row duration tracks the
    non-punctuation count (r = 0.986) far better than the raw length (r = 0.82).
    """
    return [
        i
        for i, ch in enumerate(text)
        if not unicodedata.category(ch).startswith("P") and not ch.isspace()
    ]


def preproc_vhdr(subject: str, book: str, run: int) -> Path:
    session = BOOK_TO_SESSION[book]
    return (
        DATA_ROOT
        / "derivatives"
        / "preproc"
        / "filtered_0.5_30"
        / subject
        / session
        / "eeg"
        / f"{subject}_{session}_task-reading_run-{run:02d}_eeg.vhdr"
    )


def canonical_montage_positions() -> tuple[np.ndarray, list[str]]:
    """Electrode coordinates shared by every recording in the dataset."""
    path = (
        DATA_ROOT
        / "sub-04"
        / "ses-LittlePrince"
        / "eeg"
        / "sub-04_ses-LittlePrince_space-CapTrak_electrodes.tsv"
    )
    table = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    names = table["name"].astype(str).tolist()
    xyz = table[["x", "y", "z"]].to_numpy(dtype=float)
    return xyz, names


def build_csd_matrix(out_path: Path) -> np.ndarray:
    """Extract the spherical-spline surface Laplacian as a 128x128 operator.

    CSD is linear and applied independently at each time point, so running MNE's
    implementation on an identity "epoch" recovers the operator itself.  Storing
    it lets training apply or skip CSD without a second copy of the cache.
    """
    import mne

    mne.set_log_level("ERROR")
    xyz, names = canonical_montage_positions()
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, xyz)), coord_frame="head"
    )
    info = mne.create_info(names, sfreq=TARGET_SFREQ, ch_types="eeg")
    identity = np.eye(len(names), dtype=float)[None, :, :]
    epochs = mne.EpochsArray(identity, info, verbose=False)
    epochs.set_montage(montage, verbose=False)
    csd = mne.preprocessing.compute_current_source_density(epochs, copy=True)
    matrix = csd.get_data()[0]
    np.save(out_path, matrix.astype(np.float32))
    return matrix


def read_run(subject: str, book: str, run: int) -> tuple[np.ndarray, float] | None:
    import mne

    mne.set_log_level("ERROR")
    vhdr = preproc_vhdr(subject, book, run)
    if not vhdr.exists():
        return None
    # Some GarnettDream headers reference a misspelled session token
    # (``ses-GranettDream``); the frozen builder's reader resolves that via a
    # temporary symlink directory without touching the dataset.
    raw = _read_raw_brainvision_robust(vhdr)
    raw.pick("eeg")
    if len(raw.ch_names) != N_CHANNELS:
        return None
    data = raw.get_data() * 1e6
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - SOURCE_SFREQ) > 1e-6:
        return None
    # Anti-aliased 256 -> 128 Hz.  The source is already low-passed at 30 Hz, so
    # this discards nothing but halves the cache.
    data = resample_poly(data, up=1, down=2, axis=1).astype(np.float32)
    return data, TARGET_SFREQ


def build_one(subject: str, book: str, out_dir: Path, overwrite: bool) -> dict:
    x_path = out_dir / f"X_{subject}_{book}.npy"
    meta_path = out_dir / f"meta_{subject}_{book}.csv"
    if x_path.exists() and meta_path.exists() and not overwrite:
        meta = pd.read_csv(meta_path)
        return {"subject": subject, "book": book, "status": "skipped", "n_epochs": len(meta)}

    # Reuse the row set of the frozen row-level cache so items, folds and the
    # baseline comparison stay identical.
    source = Path(os.environ.get("SURPRISAL_ROW_CACHE", ROOT / "data_cache" / "epochs")) / f"epochs_{subject}_{book}_0.5_30.npz"
    if not source.exists():
        return {"subject": subject, "book": book, "status": "missing_row_cache", "n_epochs": 0}
    row_meta = pd.DataFrame(np.load(source, allow_pickle=True)["meta"])
    row_meta["run"] = row_meta["run"].astype(int)
    row_meta["row_idx"] = row_meta["row_idx"].astype(int)
    row_meta["text"] = row_meta["text"].astype(str)

    pre = int(round(PRE_SECONDS * TARGET_SFREQ))
    post = int(round(POST_SECONDS * TARGET_SFREQ))
    width = pre + post

    chunks: list[np.ndarray] = []
    records: list[dict] = []
    n_missing_run = 0
    n_out_of_bounds = 0

    for run, run_rows in row_meta.groupby("run", sort=True):
        loaded = read_run(subject, book, int(run))
        event_path = preproc_events_path(DATA_ROOT, subject, book, int(run))
        if loaded is None or not event_path.exists():
            n_missing_run += len(run_rows)
            continue
        data, sfreq = loaded
        pairs = pair_rows_rowe(pd.read_csv(event_path, sep="\t", encoding="utf-8-sig"))
        if pairs.empty:
            n_missing_run += len(run_rows)
            continue
        pairs = pairs.set_index("row_idx")

        for _, row in run_rows.iterrows():
            row_idx = int(row["row_idx"])
            if row_idx not in pairs.index:
                n_missing_run += 1
                continue
            pair = pairs.loc[row_idx]
            text = str(row["text"])
            slots = highlight_indices(text)
            if len(slots) < 2:
                continue
            start = float(pair["start"])
            duration = float(pair["end"]) - start
            soa = duration / len(slots)

            for slot, char_pos in enumerate(slots):
                onset = int(round((start + slot * soa) * sfreq))
                lo, hi = onset - pre, onset + post
                if lo < 0 or hi > data.shape[1]:
                    n_out_of_bounds += 1
                    continue
                chunks.append(data[:, lo:hi])
                records.append(
                    {
                        "subject": subject,
                        "book": book,
                        "run": int(run),
                        "row_idx": row_idx,
                        "text": text,
                        "char": text[char_pos],
                        "char_slot": slot,
                        "n_slots": len(slots),
                        # audit-only columns; never encoder inputs
                        "n_char": int(len(text)),
                        "dur": duration,
                        "soa": soa,
                        "row_onset": start,
                    }
                )

    if not records:
        return {"subject": subject, "book": book, "status": "empty", "n_epochs": 0}

    stacked = np.stack(chunks).astype(np.float32)
    np.clip(stacked, -AMPLITUDE_CLIP_UV, AMPLITUDE_CLIP_UV, out=stacked)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_path, stacked.astype(np.float16))
    pd.DataFrame(records).to_csv(meta_path, index=False)
    return {
        "subject": subject,
        "book": book,
        "status": "built",
        "n_epochs": int(len(records)),
        "n_rows": int(row_meta.shape[0]),
        "width": int(width),
        "missing_run_rows": int(n_missing_run),
        "out_of_bounds": int(n_out_of_bounds),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=["sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
                 "sub-09", "sub-10", "sub-13", "sub-14", "sub-15"],
    )
    ap.add_argument("--books", nargs="*", default=["LittlePrince", "GarnettDream"])
    ap.add_argument("--out-dir", default=str(ROOT / "data_cache" / "char_epochs"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csd_path = out_dir / "csd_matrix.npy"
    if not csd_path.exists():
        build_csd_matrix(csd_path)
    print(f"[csd] {csd_path}", flush=True)

    jobs = [(s, b) for s in args.subjects for b in args.books]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(build_one, s, b, out_dir, args.overwrite): (s, b) for s, b in jobs
        }
        for future in as_completed(futures):
            subject, book = futures[future]
            try:
                info = future.result()
            except Exception as exc:  # noqa: BLE001
                info = {"subject": subject, "book": book, "status": f"error: {exc}"}
            results.append(info)
            print(f"[done] {info}", flush=True)

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "character-onset-locked epochs on the native physical time axis",
        "source_tier": "derivatives/preproc/filtered_0.5_30 (ICA + bad-channel "
                       "interpolation + average reference applied by dataset authors)",
        "source_sfreq_hz": SOURCE_SFREQ,
        "target_sfreq_hz": TARGET_SFREQ,
        "window_seconds": [-PRE_SECONDS, POST_SECONDS],
        "window_samples": int(round((PRE_SECONDS + POST_SECONDS) * TARGET_SFREQ)),
        "character_onset_model": "row_start + slot * (row_duration / n_highlighted_chars)",
        "baseline_correction": "not applied in cache; applied at load time",
        "epoch_normalization": "none (microvolt amplitude preserved)",
        "amplitude_clip_uv": AMPLITUDE_CLIP_UV,
        "row_set_source": "frozen row-level epoch cache (identical items and folds)",
        "audit_only_columns": ["n_char", "dur", "soa", "row_onset", "char_slot", "n_slots"],
        "results": results,
    }
    (out_dir / "CACHE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    total = sum(int(r.get("n_epochs", 0) or 0) for r in results)
    print(f"[cache] {out_dir}  total_char_epochs={total}")


if __name__ == "__main__":
    main()
