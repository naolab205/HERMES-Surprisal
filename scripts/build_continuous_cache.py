#!/usr/bin/env python
"""Continuous reading-period EEG plus a character event index, for TRF modelling.

Why continuous rather than epochs
---------------------------------
Characters are presented every ~420 ms and the N400 peaks near 400 ms, so the
response to one character is still unfolding when the next one starts. Epoch
averaging cannot separate those: every epoch contains a mixture of its own
character's response and its neighbours'. A temporal response function estimates
the response by deconvolution over the continuous signal, which is the standard
treatment in naturalistic language EEG (Heilbron et al., PNAS 2022; Crosse et
al., mTRF).

This cache keeps the concatenated reading intervals only, at 128 Hz, with an
event table giving each character's sample index inside the concatenated signal
plus its row, slot and identity. Boundaries between rows are recorded so a TRF
design matrix never convolves across a gap.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
REF_SRC = ROOT / "src"
sys.path.insert(0, str(REF_SRC))
sys.path.insert(0, str(REF_SRC.parent))

from cogalign.io.events import pair_rows_rowe, preproc_events_path  # noqa: E402
from build_char_epoch_cache import (  # noqa: E402
    AMPLITUDE_CLIP_UV, DATA_ROOT, N_CHANNELS, SOURCE_SFREQ, TARGET_SFREQ,
    highlight_indices, preproc_vhdr,
)
from scripts.build.build_epoch_cache import _read_raw_brainvision_robust  # noqa: E402


def build_one(subject: str, book: str, out_dir: Path, overwrite: bool) -> dict:
    x_path = out_dir / f"C_{subject}_{book}.npy"
    e_path = out_dir / f"events_{subject}_{book}.csv"
    if x_path.exists() and e_path.exists() and not overwrite:
        return {"subject": subject, "book": book, "status": "skipped"}

    source = Path(os.environ.get("SURPRISAL_ROW_CACHE", ROOT / "data_cache" / "epochs")) / f"epochs_{subject}_{book}_0.5_30.npz"
    if not source.exists():
        return {"subject": subject, "book": book, "status": "missing_row_cache"}
    row_meta = pd.DataFrame(np.load(source, allow_pickle=True)["meta"])
    row_meta["run"] = row_meta["run"].astype(int)
    row_meta["row_idx"] = row_meta["row_idx"].astype(int)
    row_meta["text"] = row_meta["text"].astype(str)

    import mne

    mne.set_log_level("ERROR")
    blocks: list[np.ndarray] = []
    events: list[dict] = []
    cursor = 0

    for run, run_rows in row_meta.groupby("run", sort=True):
        vhdr = preproc_vhdr(subject, book, int(run))
        event_path = preproc_events_path(DATA_ROOT, subject, book, int(run))
        if not vhdr.exists() or not event_path.exists():
            continue
        raw = _read_raw_brainvision_robust(vhdr)
        raw.pick("eeg")
        if len(raw.ch_names) != N_CHANNELS:
            continue
        sfreq = float(raw.info["sfreq"])
        if abs(sfreq - SOURCE_SFREQ) > 1e-6:
            continue
        data = resample_poly((raw.get_data() * 1e6), up=1, down=2, axis=1).astype(np.float32)
        del raw
        pairs = pair_rows_rowe(pd.read_csv(event_path, sep="\t", encoding="utf-8-sig"))
        if pairs.empty:
            continue
        pairs = pairs.set_index("row_idx")

        for _, row in run_rows.iterrows():
            row_idx = int(row["row_idx"])
            if row_idx not in pairs.index:
                continue
            pair = pairs.loc[row_idx]
            text = str(row["text"])
            slots = highlight_indices(text)
            if len(slots) < 2:
                continue
            start = float(pair["start"])
            duration = float(pair["end"]) - start
            # One SOA of lead-out so the last character's late response is
            # inside the block rather than truncated at the boundary.
            soa = duration / len(slots)
            lo = int(round(start * TARGET_SFREQ))
            hi = int(round((start + duration + soa) * TARGET_SFREQ))
            if lo < 0 or hi > data.shape[1]:
                continue
            block = np.clip(data[:, lo:hi], -AMPLITUDE_CLIP_UV, AMPLITUDE_CLIP_UV)
            blocks.append(block)
            for slot, char_pos in enumerate(slots):
                events.append(
                    {
                        "subject": subject, "book": book, "run": int(run),
                        "row_idx": row_idx, "char_slot": slot,
                        "char": text[char_pos],
                        "sample": cursor + int(round(slot * soa * TARGET_SFREQ)),
                        "block_start": cursor, "block_end": cursor + block.shape[1],
                        "n_slots": len(slots),
                        # audit-only
                        "soa": soa, "dur": duration, "n_char": len(text),
                    }
                )
            cursor += block.shape[1]

    if not blocks:
        return {"subject": subject, "book": book, "status": "empty"}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_path, np.concatenate(blocks, axis=1).astype(np.float16))
    pd.DataFrame(events).to_csv(e_path, index=False)
    return {"subject": subject, "book": book, "status": "built",
            "n_samples": int(cursor), "n_events": int(len(events)),
            "seconds": round(cursor / TARGET_SFREQ, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*",
                    default=["sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
                             "sub-09", "sub-10", "sub-13", "sub-14", "sub-15"])
    ap.add_argument("--books", nargs="*", default=["LittlePrince", "GarnettDream"])
    ap.add_argument("--out-dir", default=str(ROOT / "data_cache" / "continuous"))
    ap.add_argument("--workers", type=int, default=7)
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
        "protocol": "concatenated reading intervals, one SOA of lead-out per row",
        "sfreq_hz": TARGET_SFREQ,
        "normalization": "none (microvolts preserved)",
        "amplitude_clip_uv": AMPLITUDE_CLIP_UV,
        "purpose": "temporal response function estimation with overlap deconvolution",
        "results": results,
    }, indent=2) + "\n")
    print(f"[cache] {out_dir}")


if __name__ == "__main__":
    main()
