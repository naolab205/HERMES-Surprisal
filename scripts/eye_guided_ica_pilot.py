#!/usr/bin/env python3
"""Audit eye-event associations of supplied ChineseEEG ICA components.

This is a signal-level feasibility probe. It does not retrain ICA, remove
components, fit a surprisal decoder, or estimate held-out task performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eye_char_cache import DATA_ROOT, rar_path, read_gaze  # noqa: E402
from cogalign.core.config import BOOK_TO_SESSION  # noqa: E402
from cogalign.io.events import (  # noqa: E402
    pair_rows_rowe, preproc_events_path, raw_events_path,
)

SFREQ = 256
SACCADE_SPEED = 2.0  # normalized gaze-coordinate units per second
FIXATION_SPEED = 0.1
DEFAULT_RECORDINGS = [
    ("sub-04", "LittlePrince", 1),
    ("sub-04", "GarnettDream", 1),
    ("sub-05", "GarnettDream", 1),
    ("sub-05", "LittlePrince", 1),
    ("sub-06", "LittlePrince", 1),
    ("sub-07", "LittlePrince", 1),
    ("sub-08", "GarnettDream", 1),
]


def mark_samples(mask: np.ndarray, centers: np.ndarray, radius: int) -> None:
    for center in np.rint(centers * SFREQ).astype(int):
        if radius <= center < len(mask) - radius:
            mask[center - radius:center + radius + 1] = True


def within_rows(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(starts, times, side="right") - 1
    return (pos >= 0) & (times < ends[np.clip(pos, 0, len(ends) - 1)])


def audit_one(subject: str, book: str, run: int) -> tuple[dict, list[dict]]:
    session = BOOK_TO_SESSION[book]
    folder = (DATA_ROOT / "derivatives/preproc/filtered_0.5_30" /
              subject / session / "eeg")
    stem = f"{subject}_{session}_task-reading_run-{run:02d}"
    ic_path = folder / f"{stem}_ica_components.npy"
    excluded_path = folder / f"{stem}_ica_components.json"
    if not ic_path.exists() or not excluded_path.exists():
        raise FileNotFoundError(f"Missing ICA artifact for {stem}")
    components = np.load(ic_path, mmap_mode="r")
    if components.ndim != 2 or components.shape[0] != 20:
        raise ValueError(f"Unexpected ICA shape {components.shape}: {ic_path}")
    excluded = set(json.loads(excluded_path.read_text())["exclude"])

    raw_events = pd.read_csv(raw_events_path(DATA_ROOT, subject, book, run), sep="\t")
    pre_events = pd.read_csv(preproc_events_path(DATA_ROOT, subject, book, run), sep="\t")
    raw_rows = pair_rows_rowe(raw_events)
    pre_rows = pair_rows_rowe(pre_events)
    if raw_rows.empty or pre_rows.empty:
        raise ValueError(f"No row pairs for {stem}")
    eyes = raw_events.loc[raw_events.trial_type.eq("EYES"), "onset"].to_numpy()
    if len(eyes) != 1:
        raise ValueError(f"Expected one EYES marker for {stem}; got {len(eyes)}")
    shift = float(raw_rows.start.iloc[0] - pre_rows.start.iloc[0] - eyes[0])
    loaded = read_gaze(rar_path(subject, book, run))
    if loaded is None:
        raise FileNotFoundError(f"Missing readable eye archive for {stem}")
    times, eye = loaded
    starts, ends = pre_rows.start.to_numpy(), pre_rows.end.to_numpy()
    n_samples = components.shape[1]
    reading = np.zeros(n_samples, dtype=bool)
    for start, end in zip(starts, ends):
        lo, hi = int(max(0, round(start * SFREQ))), int(min(n_samples, round(end * SFREQ)))
        reading[lo:hi] = True

    dt = np.diff(times)
    displacement = np.linalg.norm(np.diff(eye[:, :2], axis=0), axis=1)
    valid = (eye[:-1, 3] > .5) & (eye[1:, 3] > .5)
    valid &= np.isfinite(displacement) & (dt > .015) & (dt < .025)
    speed = np.divide(displacement, dt, out=np.full_like(displacement, np.nan),
                      where=valid)
    mid = (times[:-1] + times[1:]) / 2 - shift
    in_row = within_rows(mid, starts, ends)
    saccade = valid & in_row & (speed >= SACCADE_SPEED)
    fixation = valid & in_row & (speed <= FIXATION_SPEED)
    saccade_samples = np.zeros(n_samples, dtype=bool)
    fixation_samples = np.zeros(n_samples, dtype=bool)
    mark_samples(saccade_samples, mid[saccade], radius=10)  # +/-39 ms
    mark_samples(fixation_samples, mid[fixation], radius=4)  # +/-16 ms
    saccade_samples &= reading
    fixation_samples &= reading & ~saccade_samples

    invalid = eye[:, 3] <= .5
    gap_start = np.flatnonzero(invalid & np.r_[True, ~invalid[:-1]])
    gap_end = np.flatnonzero(invalid & np.r_[~invalid[1:], True])
    gap_center = (times[gap_start] + times[gap_end]) / 2 - shift
    gap_length = gap_end - gap_start + 1
    gap_keep = (gap_length >= 3) & (gap_length <= 40)
    gap_keep &= within_rows(gap_center, starts, ends)
    blink_samples = np.zeros(n_samples, dtype=bool)
    mark_samples(blink_samples, gap_center[gap_keep], radius=40)  # +/-156 ms
    blink_samples &= reading
    blink_reference = reading & ~blink_samples
    counts = [saccade_samples.sum(), fixation_samples.sum(),
              blink_samples.sum(), blink_reference.sum()]
    if min(counts) < 1000:
        raise ValueError(f"Too few eye-event EEG samples in {stem}: {counts}")

    saccade_var = np.var(components[:, saccade_samples], axis=1)
    fixation_var = np.var(components[:, fixation_samples], axis=1)
    blink_var = np.var(components[:, blink_samples], axis=1)
    reference_var = np.var(components[:, blink_reference], axis=1)
    saccade_ratio = saccade_var / np.maximum(fixation_var, 1e-12)
    blink_ratio = blink_var / np.maximum(reference_var, 1e-12)
    rows = [{"subject": subject, "book": book, "run": run,
             "component": i, "author_excluded": i in excluded,
             "saccade_fixation_variance_ratio": float(saccade_ratio[i]),
             "invalid_gap_variance_ratio": float(blink_ratio[i])}
            for i in range(components.shape[0])]
    info = {"subject": subject, "book": book, "run": run,
            "ica_path": str(ic_path), "eye_archive": str(rar_path(subject, book, run)),
            "eye_to_preproc_shift_seconds": shift,
            "eye_valid_fraction": float(np.mean(eye[:, 3] > .5)),
            "author_excluded_components": sorted(excluded),
            "n_gaze_saccade_frames": int(saccade.sum()),
            "n_gaze_fixation_frames": int(fixation.sum()),
            "n_blink_like_invalid_gaps": int(gap_keep.sum()),
            "n_ica_samples_saccade": int(counts[0]),
            "n_ica_samples_fixation": int(counts[1]),
            "n_ica_samples_invalid_gap": int(counts[2]),
            "n_ica_samples_reference": int(counts[3]),
            "top_saccade_components": np.argsort(saccade_ratio)[-3:][::-1].tolist(),
            "top_invalid_gap_components": np.argsort(blink_ratio)[-3:][::-1].tolist()}
    return info, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results/eye_guided_ica_pilot_20260930")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports, rows = [], []
    for recording in DEFAULT_RECORDINGS:
        report, component_rows = audit_one(*recording)
        reports.append(report)
        rows.extend(component_rows)
        print(f"{recording}: excluded={report['author_excluded_components']} "
              f"top_saccade={report['top_saccade_components']} "
              f"top_invalid_gap={report['top_invalid_gap_components']}", flush=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "component_event_variance.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps({
        "status": "signal_level_feasibility_only",
        "saccade_speed_threshold_normalized_per_second": SACCADE_SPEED,
        "fixation_speed_threshold_normalized_per_second": FIXATION_SPEED,
        "invalid_gap_length_samples": [3, 40],
        "recordings": reports,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
