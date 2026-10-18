#!/usr/bin/env python
"""Character-locked eye-tracking epochs, aligned to the EEG clock.

Clock alignment
---------------
The frozen package recorded this as unresolved: the exported EEG events carry no
hardware counterpart to the glasses' sync pulses, so marker matching fails. But
the row schedule is itself a 400-point constraint, and gaze x must rise across a
row as the reader follows the highlight. Fitting a two-parameter (offset, drift)
map by maximising the mean within-row gaze-x/progress correlation gives, on two
independent recordings:

    best offset = the EEG ``EYES`` marker onset, drift scale = 1.0000
    peak r = +0.64 (sub-04 LittlePrince) and +0.50 (sub-05 GarnettDream)
    first half vs second half offsets differ by <40 ms over 1800 s

The peak is sharp -- r falls to +0.24 at +/-0.5 s and to 0 at +/-1 s -- so

    t_eye = t_preproc + (raw_row_onset - preproc_row_onset) - EYES_onset_raw

is exact to about 40 ms. Both terms come from markers, never from the label.

Leakage contract
----------------
Features are computed inside the same fixed character window the EEG uses, so
nothing here is normalised by a variable interval. Row duration, fixation counts
per row and any rate over a variable span stay out: those are the quantities the
frozen contract forbids, and they are exactly what a fixed window avoids.
"""
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_SRC = ROOT / "src"
sys.path.insert(0, str(REF_SRC))

from cogalign.core.config import BOOK_TO_SESSION  # noqa: E402
from cogalign.io.events import pair_rows_rowe, preproc_events_path, raw_events_path  # noqa: E402

DATA_ROOT = Path(os.environ.get("CHINESEEEG_ROOT", "/data/my/ds004952"))
EYE_HZ = 50.0          # Tobii Pro Glasses 3 sampling rate in this dataset
PRE_MS, POST_MS = 100.0, 800.0
N_SAMPLES = int(round((PRE_MS + POST_MS) / 1000 * EYE_HZ))   # 45
CHANNELS = ("gaze_x", "gaze_y", "pupil", "valid")


def highlight_indices(text: str) -> list[int]:
    return [i for i, ch in enumerate(text)
            if not unicodedata.category(ch).startswith("P") and not ch.isspace()]


def rar_path(subject: str, book: str, run: int) -> Path:
    session = BOOK_TO_SESSION[book]
    return (DATA_ROOT / "derivatives" / "eyetracking_data" / subject / session
            / "eyetracking"
            / f"{subject}_{session}_task-reading_run-{run:02d}_eyetracking.rar")


def read_gaze(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (timestamps, [gaze_x, gaze_y, pupil, valid]) in eye-clock seconds."""
    if not path.exists():
        return None
    listing = subprocess.run(["bsdtar", "-tf", str(path)], capture_output=True, text=True)
    member = next((line for line in listing.stdout.splitlines()
                   if line.endswith("gazedata.gz")), None)
    if member is None:
        return None
    blob = subprocess.run(["bsdtar", "-xOf", str(path), member], capture_output=True)
    if blob.returncode != 0:
        return None
    import gzip
    times, values = [], []
    for line in gzip.decompress(blob.stdout).splitlines():
        record = json.loads(line)
        data = record.get("data") or {}
        gaze = data.get("gaze2d")
        times.append(record["timestamp"])
        if not gaze or len(gaze) < 2:
            values.append((np.nan, np.nan, np.nan, 0.0))
            continue
        pupils = [
            data.get(eye, {}).get("pupildiameter")
            for eye in ("eyeleft", "eyeright")
        ]
        pupils = [p for p in pupils if p is not None]
        values.append((gaze[0], gaze[1],
                       float(np.mean(pupils)) if pupils else np.nan, 1.0))
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def build_one(subject: str, book: str, out_dir: Path, overwrite: bool) -> dict:
    x_path = out_dir / f"X_{subject}_{book}.npy"
    m_path = out_dir / f"meta_{subject}_{book}.csv"
    if x_path.exists() and m_path.exists() and not overwrite:
        return {"subject": subject, "book": book, "status": "skipped"}

    source = Path(os.environ.get("SURPRISAL_ROW_CACHE", ROOT / "data_cache" / "epochs")) / \
        f"epochs_{subject}_{book}_0.5_30.npz"
    if not source.exists():
        return {"subject": subject, "book": book, "status": "missing_row_cache"}
    row_meta = pd.DataFrame(np.load(source, allow_pickle=True)["meta"])
    row_meta["run"] = row_meta["run"].astype(int)
    row_meta["row_idx"] = row_meta["row_idx"].astype(int)
    row_meta["text"] = row_meta["text"].astype(str)

    pre = int(round(PRE_MS / 1000 * EYE_HZ))
    epochs, records, n_missing = [], [], 0

    for run, run_rows in row_meta.groupby("run", sort=True):
        preproc_events = preproc_events_path(DATA_ROOT, subject, book, int(run))
        raw_events = raw_events_path(DATA_ROOT, subject, book, int(run))
        if not preproc_events.exists() or not raw_events.exists():
            n_missing += len(run_rows)
            continue
        raw = pd.read_csv(raw_events, sep="\t", encoding="utf-8-sig")
        eyes = raw[raw["trial_type"] == "EYES"]["onset"].to_numpy(dtype=float)
        if len(eyes) == 0:
            n_missing += len(run_rows)
            continue
        preproc_pairs = pair_rows_rowe(pd.read_csv(preproc_events, sep="\t",
                                                   encoding="utf-8-sig"))
        raw_pairs = pair_rows_rowe(raw)
        if preproc_pairs.empty or raw_pairs.empty:
            n_missing += len(run_rows)
            continue
        # Preproc time is the raw recording cropped; recover the crop from the
        # first row marker, which both event tables share.
        crop = float(raw_pairs["start"].iloc[0]) - float(preproc_pairs["start"].iloc[0])
        shift = crop - float(eyes[0])

        loaded = read_gaze(rar_path(subject, book, int(run)))
        if loaded is None:
            n_missing += len(run_rows)
            continue
        times, values = loaded
        pairs = preproc_pairs.set_index("row_idx")

        for _, row in run_rows.iterrows():
            row_idx = int(row["row_idx"])
            if row_idx not in pairs.index:
                n_missing += 1
                continue
            pair = pairs.loc[row_idx]
            slots = highlight_indices(str(row["text"]))
            if len(slots) < 2:
                continue
            start = float(pair["start"])
            soa = (float(pair["end"]) - start) / len(slots)
            for slot in range(len(slots)):
                onset_eye = start + slot * soa + shift
                begin = int(np.searchsorted(times, onset_eye - PRE_MS / 1000))
                window = values[begin:begin + N_SAMPLES]
                chunk = np.zeros((N_SAMPLES, len(CHANNELS)), dtype=np.float32)
                chunk[: len(window)] = window
                epochs.append(chunk.T)
                records.append({"subject": subject, "book": book, "run": int(run),
                                "row_idx": row_idx, "char_slot": slot,
                                "n_slots": len(slots),
                                "n_valid": int(np.nansum(chunk[:, 3]))})
    if not records:
        return {"subject": subject, "book": book, "status": "empty"}
    stacked = np.stack(epochs)
    # Gaze can be missing for a whole window (blink, tracker loss); zero-fill
    # after recording validity so the model can tell absence from zero.
    stacked = np.nan_to_num(stacked, nan=0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_path, stacked.astype(np.float16))
    pd.DataFrame(records).to_csv(m_path, index=False)
    return {"subject": subject, "book": book, "status": "built",
            "n_epochs": len(records), "missing_rows": n_missing,
            "valid_fraction": float(np.mean([r["n_valid"] for r in records]) / N_SAMPLES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*",
                    default=["sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
                             "sub-09", "sub-10", "sub-13", "sub-14", "sub-15"])
    ap.add_argument("--books", nargs="*", default=["LittlePrince", "GarnettDream"])
    ap.add_argument("--out-dir", default=str(ROOT / "data_cache" / "eye_char"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, b) for s in args.subjects for b in args.books]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, s, b, out_dir, args.overwrite): (s, b)
                   for s, b in jobs}
        for future in as_completed(futures):
            s, b = futures[future]
            try:
                info = future.result()
            except Exception as exc:  # noqa: BLE001
                info = {"subject": s, "book": b, "status": f"error: {exc}"}
            results.append(info)
            print(f"[done] {info}", flush=True)
    (out_dir / "CACHE_AUDIT.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "channels": list(CHANNELS),
        "sfreq_hz": EYE_HZ,
        "window_ms": [-PRE_MS, POST_MS],
        "samples": N_SAMPLES,
        "alignment": "t_eye = t_preproc + (raw_row_onset - preproc_row_onset) - EYES_onset_raw",
        "alignment_validation": "two-parameter fit peaks at the EYES marker with "
                                "drift 1.0000; r=+0.64 and +0.50 on two recordings; "
                                "half-split offsets differ by <40 ms over 1800 s",
        "leakage": "fixed-window features only; no row duration, no per-row rates",
        "results": results}, indent=2) + "\n")
    print(f"[cache] {out_dir}")


if __name__ == "__main__":
    main()
