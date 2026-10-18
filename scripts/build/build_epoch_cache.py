#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cogalign.core.config import BOOK_TO_SESSION, ExperimentConfig, Paths
from cogalign.io.events import pair_rows_rowe, preproc_events_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build new CogAlign EEG epoch cache from ChineseEEG derivatives.")
    ap.add_argument("--subjects", nargs="*", default=list(ExperimentConfig().subjects))
    ap.add_argument("--books", nargs="*", default=list(ExperimentConfig().books))
    ap.add_argument("--data-root", default="/data/my/ds004952")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def preproc_vhdr(data_root: Path, subject: str, book: str, run: int) -> Path:
    session = BOOK_TO_SESSION[book]
    return (
        data_root
        / "derivatives"
        / "preproc"
        / "filtered_0.5_30"
        / subject
        / session
        / "eeg"
        / f"{subject}_{session}_task-reading_run-{run:02d}_eeg.vhdr"
    )


def list_runs(data_root: Path, subject: str, book: str) -> list[int]:
    session = BOOK_TO_SESSION[book]
    root = data_root / "derivatives" / "preproc" / "filtered_0.5_30" / subject / session / "eeg"
    runs = []
    for path in sorted(glob.glob(str(root / "*task-reading_run-*_eeg.vhdr"))):
        runs.append(int(Path(path).name.split("run-")[1][:2]))
    return runs


def text_rows(data_root: Path, book: str, run: int) -> list[str]:
    path = (
        data_root
        / "derivatives"
        / "novels"
        / "segmented_novel"
        / book
        / f"segmented_Chinense_novel_run_{run}.xlsx"
    )
    if not path.exists():
        return []
    df = pd.read_excel(path)
    rows = []
    for _, row in df.iterrows():
        text = str(row.iloc[0]) if len(row) else ""
        rows.append(text.strip() if text and text != "nan" else "")
    return rows


def text_embedding(data_root: Path, book: str, run: int) -> np.ndarray | None:
    path = data_root / "derivatives" / "text_embeddings" / f"{book}_text_embedding" / f"text_embedding_run_{run}.npy"
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def _read_raw_brainvision_robust(vhdr: Path):
    """Read a BrainVision .vhdr, tolerating internal DataFile/MarkerFile typos.

    Some ChineseEEG GarnettDream derivatives ship .vhdr/.vmrk headers that point
    to a misspelled session token (e.g. ``ses-GranettDream``) while the actual
    files on disk use the correct ``ses-GarnettDream`` spelling. When the
    referenced sibling files are missing, build a temporary directory of
    symlinks named exactly as the headers expect, pointing back to the real
    files, and read from there. No raw dataset files are modified.
    """
    import mne

    text = vhdr.read_text(encoding="utf-8", errors="ignore")
    refs = {}
    for key in ("DataFile", "MarkerFile"):
        m = re.search(rf"^{key}=(.+)$", text, flags=re.MULTILINE)
        if m:
            refs[key] = m.group(1).strip()

    # If every referenced sibling already exists, read directly.
    if all((vhdr.parent / name).exists() for name in refs.values()):
        return mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

    # Real sibling files share the vhdr's actual basename (minus extension).
    real_base = vhdr.name[: -len(".vhdr")]
    ext_to_real = {
        ".eeg": vhdr.parent / f"{real_base}.eeg",
        ".vmrk": vhdr.parent / f"{real_base}.vmrk",
    }
    tmp = Path(tempfile.mkdtemp(prefix="vhdr_fix_"))
    try:
        # The vhdr itself: keep its referenced/own name in the temp dir.
        os.symlink(vhdr, tmp / vhdr.name)
        for name in refs.values():
            ext = Path(name).suffix
            real = ext_to_real.get(ext)
            if real is None or not real.exists():
                raise FileNotFoundError(f"cannot resolve referenced file {name} for {vhdr}")
            link = tmp / name
            if not link.exists():
                os.symlink(real, link)
        return mne.io.read_raw_brainvision(tmp / vhdr.name, preload=True, verbose=False)
    finally:
        # Cleanup happens after MNE preloads data into memory above.
        for child in tmp.glob("*"):
            child.unlink()
        tmp.rmdir()


def build_one(data_root: Path, out_dir: Path, subject: str, book: str, overwrite: bool) -> Path | None:
    out = out_dir / f"epochs_{subject}_{book}_0.5_30.npz"
    if out.exists() and not overwrite:
        print(f"[skip] {out}")
        return out

    import mne

    mne.set_log_level("ERROR")
    eeg_rows = []
    text_embeds = []
    meta_rows = []
    for run in list_runs(data_root, subject, book):
        vhdr = preproc_vhdr(data_root, subject, book, run)
        ev_path = preproc_events_path(data_root, subject, book, run)
        emb = text_embedding(data_root, book, run)
        rows = text_rows(data_root, book, run)
        if emb is None or not rows or not vhdr.exists() or not ev_path.exists():
            continue
        try:
            raw = _read_raw_brainvision_robust(vhdr)
        except Exception as exc:
            print(f"[warn] failed {subject} {book} run-{run:02d}: {exc}")
            continue
        raw.pick("eeg")
        data = raw.get_data() * 1e6
        sfreq = float(raw.info["sfreq"])
        pairs = pair_rows_rowe(pd.read_csv(ev_path, sep="\t", encoding="utf-8-sig"))
        n_text = min(len(rows), len(emb))
        kept = 0
        for _, pair in pairs.iterrows():
            row_idx = int(pair["row_idx"])
            if row_idx >= n_text:
                continue
            text = rows[row_idx]
            if not text:
                continue
            s0 = max(0, int(float(pair["start"]) * sfreq))
            s1 = min(data.shape[1], int(float(pair["end"]) * sfreq))
            if s1 - s0 < int(0.2 * sfreq):
                continue
            seg = resample(data[:, s0:s1], 128, axis=1).astype(np.float32)
            med = np.median(seg, axis=1, keepdims=True)
            mad = np.median(np.abs(seg - med), axis=1, keepdims=True) + 1e-6
            seg = (seg - med) / (1.4826 * mad)
            eeg_rows.append(seg.astype(np.float16))
            text_embeds.append(emb[row_idx].astype(np.float32))
            meta_rows.append(
                {
                    "subject": subject,
                    "book": book,
                    "run": run,
                    "row_idx": row_idx,
                    "text": text,
                    "n_char": len(text),
                    "dur": float(pair["dur"]),
                }
            )
            kept += 1
        print(f"[run] {subject} {book} run-{run:02d}: {kept}")
    if not eeg_rows:
        print(f"[warn] no rows for {subject} {book}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.stack(eeg_rows),
        E=np.vstack(text_embeds),
        meta=pd.DataFrame(meta_rows).to_records(index=False),
        n_ch=np.array(128),
    )
    print(f"[save] {out}")
    return out


def main() -> None:
    args = parse_args()
    paths = Paths(project_root=ROOT, data_root=Path(args.data_root))
    paths.epoch_cache.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        for book in args.books:
            build_one(paths.data_root, paths.epoch_cache, subject, book, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
