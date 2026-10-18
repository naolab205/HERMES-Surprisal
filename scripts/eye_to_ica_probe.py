#!/usr/bin/env python3
"""Time-blocked eye-to-ICA prediction on supplied continuous recordings.

The ridge model uses eye data only and predicts author-provided ICA activations.
It is a signal-level feasibility probe, not an EEG correction or surprisal test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eye_char_cache import DATA_ROOT, rar_path, read_gaze  # noqa: E402
from cogalign.core.config import BOOK_TO_SESSION  # noqa: E402
from cogalign.io.events import pair_rows_rowe, raw_events_path, preproc_events_path  # noqa: E402
from eye_guided_ica_pilot import DEFAULT_RECORDINGS, SFREQ  # noqa: E402

TRAIN_FRACTION = .6
RIDGE_ALPHA = 100.0
LAGS = (-2, -1, 0, 1, 2)  # 50-Hz eye frames, +/-40 ms
MISMATCH_SHIFT_FRAMES = 500  # approximately 10 seconds of reading eye frames


def probe_one(subject: str, book: str, run: int) -> tuple[dict, list[dict]]:
    session = BOOK_TO_SESSION[book]
    stem = f"{subject}_{session}_task-reading_run-{run:02d}"
    folder = (DATA_ROOT / "derivatives/preproc/filtered_0.5_30" /
              subject / session / "eeg")
    ic_path = folder / f"{stem}_ica_components.npy"
    ic = np.load(ic_path, mmap_mode="r")
    excluded = set(json.loads(ic_path.with_suffix(".json").read_text())["exclude"])
    raw = pd.read_csv(raw_events_path(DATA_ROOT, subject, book, run), sep="\t")
    pre = pd.read_csv(preproc_events_path(DATA_ROOT, subject, book, run), sep="\t")
    raw_rows, pre_rows = pair_rows_rowe(raw), pair_rows_rowe(pre)
    eyes = raw.loc[raw.trial_type.eq("EYES"), "onset"].to_numpy()
    if len(eyes) != 1 or raw_rows.empty or pre_rows.empty:
        raise ValueError(f"Cannot establish eye/EEG alignment for {stem}")
    shift = float(raw_rows.start.iloc[0] - pre_rows.start.iloc[0] - eyes[0])
    loaded = read_gaze(rar_path(subject, book, run))
    if loaded is None:
        raise FileNotFoundError(f"Eye archive for {stem}")
    eye_time, eye = loaded
    eeg_time = eye_time - shift
    starts, ends = pre_rows.start.to_numpy(), pre_rows.end.to_numpy()
    pos = np.searchsorted(starts, eeg_time, side="right") - 1
    reading = (pos >= 0) & (eeg_time < ends[np.clip(pos, 0, len(ends) - 1)])
    reading &= (eeg_time >= 0) & (eeg_time < ic.shape[1] / SFREQ)

    base = np.nan_to_num(eye, nan=0.0)[:, [0, 1, 2, 3]]
    delta = np.diff(base[:, :2], axis=0, prepend=base[:1, :2])
    base = np.column_stack((base, delta))
    features = np.concatenate([np.roll(base, lag, axis=0) for lag in LAGS], axis=1)
    indices = np.flatnonzero(reading)
    if len(indices) < 10000:
        raise ValueError(f"Too few reading eye frames for {stem}: {len(indices)}")
    order = np.argsort(eeg_time[indices])
    indices = indices[order]
    x_raw = features[indices]
    t = eeg_time[indices]
    sample_time = np.arange(ic.shape[1]) / SFREQ
    y = np.column_stack([np.interp(t, sample_time, ic[k]) for k in range(ic.shape[0])])
    n_train = int(len(x_raw) * TRAIN_FRACTION)

    def fit_and_score(design: np.ndarray) -> np.ndarray:
        mean, std = design[:n_train].mean(axis=0), design[:n_train].std(axis=0)
        std[std < 1e-6] = 1.0
        standardized = (design - mean) / std
        model = Ridge(alpha=RIDGE_ALPHA).fit(standardized[:n_train], y[:n_train])
        return r2_score(y[n_train:], model.predict(standardized[n_train:]),
                        multioutput="raw_values")

    scores = fit_and_score(x_raw)
    mismatched = fit_and_score(np.roll(x_raw, MISMATCH_SHIFT_FRAMES, axis=0))
    rows = [{"subject": subject, "book": book, "run": run,
             "component": k, "author_excluded": k in excluded,
             "eye_to_ica_test_r2": float(scores[k]),
             "mismatched_eye_test_r2": float(mismatched[k])}
            for k in range(ic.shape[0])]
    report = {"subject": subject, "book": book, "run": run,
              "n_train_eye_frames": n_train,
              "n_test_eye_frames": len(x_raw) - n_train,
              "train_end_eeg_seconds": float(t[n_train - 1]),
              "test_start_eeg_seconds": float(t[n_train]),
              "author_excluded_components": sorted(excluded),
              "top_eye_predictable_components": np.argsort(scores)[-3:][::-1].tolist(),
              "max_test_r2": float(np.max(scores)),
              "max_mismatched_test_r2": float(np.max(mismatched))}
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results/eye_guided_ica_pilot_20260930")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports, rows = [], []
    for recording in DEFAULT_RECORDINGS:
        report, component_rows = probe_one(*recording)
        reports.append(report)
        rows.extend(component_rows)
        print(f"{recording}: excluded={report['author_excluded_components']} "
              f"top={report['top_eye_predictable_components']} "
              f"max_test_r2={report['max_test_r2']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "eye_to_ica_test_r2.csv", index=False)
    (args.out_dir / "eye_to_ica_audit.json").write_text(json.dumps({
        "status": "time_blocked_signal_probe_only_no_surprisal_training",
        "train_fraction": TRAIN_FRACTION,
        "ridge_alpha": RIDGE_ALPHA,
        "eye_frame_lags": LAGS,
        "mismatched_eye_shift_frames": MISMATCH_SHIFT_FRAMES,
        "recordings": reports,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
