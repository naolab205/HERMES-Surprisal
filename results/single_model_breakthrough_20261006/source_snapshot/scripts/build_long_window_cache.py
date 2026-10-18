#!/usr/bin/env python
"""Re-cut character epochs with a window that covers the whole measured response.

The original cache stops at +450 ms. The surprisal temporal response function
estimated on this corpus peaks at 367 ms but carries a sustained negativity of
comparable size out to at least 790 ms (-0.16 at 742 ms against -0.18 at the
peak), so roughly half of the modulated interval was being discarded.

Extending the window also admits the following characters' responses, which an
epoch model cannot deconvolve. Whether the extra signal outweighs that overlap
is an empirical question, which is the point of building this.

Cut from the existing continuous cache rather than from raw BrainVision: the
continuous tier already holds the concatenated reading intervals at 128 Hz with
one SOA of lead-out per row, so this is pure indexing.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONT = ROOT / "data_cache" / "continuous"
SFREQ = 128.0


def build_one(stem: str, out_dir: Path, pre_ms: float, post_ms: float,
              overwrite: bool) -> dict:
    raise RuntimeError(
        "Spliced row intervals cannot supply complete physical EEG windows; "
        "use rebuild_physical_char_cache.py."
    )
    x_path = out_dir / f"X_{stem}.npy"
    m_path = out_dir / f"meta_{stem}.csv"
    if x_path.exists() and m_path.exists() and not overwrite:
        return {"recording": stem, "status": "skipped"}
    events = pd.read_csv(CONT / f"events_{stem}.csv")
    signal = np.load(CONT / f"C_{stem}.npy", mmap_mode="r")
    pre = int(round(pre_ms / 1000 * SFREQ))
    post = int(round(post_ms / 1000 * SFREQ))
    total = signal.shape[1]

    keep, out = [], []
    for row in events.itertuples():
        lo, hi = row.sample - pre, row.sample + post
        # Pad rather than drop when the tail runs past the block: the lead-out is
        # one SOA, so a long window always overruns the final characters of a row
        # and dropping them would bias the bag towards early slots.
        if lo < 0 or lo >= total:
            continue
        chunk = np.zeros((signal.shape[0], pre + post), dtype=np.float32)
        take = min(hi, total) - lo
        chunk[:, :take] = signal[:, lo:lo + take]
        out.append(chunk)
        keep.append(row.Index)
    if not out:
        return {"recording": stem, "status": "empty"}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_path, np.stack(out).astype(np.float16))
    meta = events.loc[keep, ["subject", "book", "run", "row_idx", "char_slot",
                             "char", "n_slots", "n_char", "dur", "soa"]].copy()
    meta["text"] = ""
    meta.to_csv(m_path, index=False)
    return {"recording": stem, "status": "built", "n_epochs": len(out)}


def main() -> None:
    raise SystemExit(
        "This legacy builder extracts across spliced row intervals and cannot "
        "produce valid long physical-time windows. Use "
        "scripts/rebuild_physical_char_cache.py for the audited 1200ms cache. "
        "Existing caches are preserved; no files were changed."
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", type=float, default=100.0)
    ap.add_argument("--post-ms", type=float, default=800.0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir or ROOT / "data_cache" / f"char_epochs_{int(args.post_ms)}ms")
    results = []
    for path in sorted(CONT.glob("events_*.csv")):
        stem = path.name[len("events_"):-len(".csv")]
        info = build_one(stem, out_dir, args.pre_ms, args.post_ms, args.overwrite)
        results.append(info)
        print(f"[done] {info}", flush=True)
    # Row text lives in the original cache; item ids are built from it.
    text = pd.concat([pd.read_csv(f, usecols=["book", "run", "row_idx", "text"])
                      for f in (ROOT / "data_cache" / "char_epochs").glob("meta_*.csv")],
                     ignore_index=True).drop_duplicates(["book", "run", "row_idx"])
    for path in sorted(out_dir.glob("meta_*.csv")):
        meta = pd.read_csv(path).drop(columns=["text"])
        meta = meta.merge(text, on=["book", "run", "row_idx"], how="left")
        meta.to_csv(path, index=False)
    (out_dir / "csd_matrix.npy").write_bytes(
        (ROOT / "data_cache" / "char_epochs" / "csd_matrix.npy").read_bytes())
    (out_dir / "CACHE_AUDIT.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "window_ms": [-args.pre_ms, args.post_ms],
        "samples": int(round((args.pre_ms + args.post_ms) / 1000 * SFREQ)),
        "source": "data_cache/continuous (pure re-indexing)",
        "rationale": "surprisal TRF carries a sustained component to ~790 ms",
        "tail_policy": "zero padded past the block end, never dropped",
        "results": results,
    }, indent=2) + "\n")
    print(f"[cache] {out_dir}")


if __name__ == "__main__":
    main()
