#!/usr/bin/env python3
"""Descriptive, measured high-minus-low row-surprisal EEG scalp maps.

Uses the ORIGINAL +450 ms character epochs, avoiding the later long-window
cache's zero-padded tails. Rows are balanced by averaging characters within
row, contrasting only 10-slot rows, then averaging the two book contrasts
within each complete reader. These maps are sensor-voltage associations, not
MM_mid attribution maps or evidence of a causal surprisal response.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = Path('/data/2_readEEG/superisal_code')
EPOCHS = FOUNDATION / 'data_cache/char_epochs'
LABELS = ROOT / 'data_cache/surprisal_1829a99ad4.csv'
ELECTRODES = Path('/data/my/ds004952/sub-04/ses-LittlePrince/eeg/sub-04_ses-LittlePrince_space-CapTrak_electrodes.tsv')
OUTPUT = ROOT / 'outputs/paper_figures/fig04_measured_surprisal_topomaps.pdf'
AUDIT = ROOT / 'analysis/real_topomap_audit.json'
BOOKS = ('GarnettDream', 'LittlePrince')
SUBJECTS = ('sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-13', 'sub-14')
SFREQ = 128.0
WINDOWS_MS = ((0, 100), (100, 200), (200, 300), (300, 400))
CHUNK = 2048


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thresholds(label: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for book in BOOKS:
        meta = pd.read_csv(EPOCHS / f'meta_sub-04_{book}.csv', usecols=['run', 'row_idx', 'n_slots'])
        rows = meta.loc[meta.n_slots.eq(10)].drop_duplicates(['run', 'row_idx'])
        rows = rows.merge(label.loc[label.book.eq(book), ['run', 'row_idx', 'surprisal']], on=['run', 'row_idx'], validate='one_to_one')
        assert rows.surprisal.notna().all() and len(rows) > 1000
        out[book] = tuple(float(v) for v in rows.surprisal.quantile([1 / 3, 2 / 3]).to_numpy())
    return out


def one_recording(subject: str, book: str, labels: pd.DataFrame,
                  cut: tuple[float, float]) -> tuple[np.ndarray, dict]:
    xp = EPOCHS / f'X_{subject}_{book}.npy'
    mp = EPOCHS / f'meta_{subject}_{book}.csv'
    x = np.load(xp, mmap_mode='r')
    meta = pd.read_csv(mp, usecols=['run', 'row_idx', 'n_slots'])
    assert x.ndim == 3 and x.shape[0] == len(meta) and x.shape[1:] == (128, 71)
    row_index = pd.MultiIndex.from_frame(meta[['run', 'row_idx']])
    code, _ = pd.factorize(row_index, sort=False)
    first = ~row_index.duplicated()
    row_info = meta.loc[first].copy()
    assert len(row_info) == int(code.max()) + 1
    row_info = row_info.merge(labels.loc[labels.book.eq(book), ['run', 'row_idx', 'surprisal']],
                              on=['run', 'row_idx'], how='left', validate='one_to_one')
    assert row_info.surprisal.notna().all()
    qlo, qhi = cut
    cond = np.full(len(row_info), -1, dtype=np.int8)
    eligible = row_info.n_slots.to_numpy() == 10
    vals = row_info.surprisal.to_numpy()
    cond[eligible & (vals <= qlo)] = 0
    cond[eligible & (vals >= qhi)] = 1
    row_counts = np.bincount(code)
    weights = 1 / row_counts[code]
    event_cond = cond[code]
    sample_time = -100.0 + 1000.0 * np.arange(71) / SFREQ
    baseline_idx = np.flatnonzero((sample_time >= -100) & (sample_time < 0))
    window_idx = [np.flatnonzero((sample_time >= lo) & (sample_time < hi)) for lo, hi in WINDOWS_MS]
    sums = np.zeros((2, len(WINDOWS_MS), 128), dtype=np.float64)
    for start in range(0, len(x), CHUNK):
        stop = min(start + CHUNK, len(x))
        sel = event_cond[start:stop] >= 0
        if not sel.any():
            continue
        arr = np.asarray(x[start:stop], dtype=np.float32)[sel]
        base = arr[:, :, baseline_idx].mean(axis=2)
        ec = event_cond[start:stop][sel]
        ew = weights[start:stop][sel]
        for wi, ii in enumerate(window_idx):
            v = arr[:, :, ii].mean(axis=2) - base
            for c in (0, 1):
                keep = ec == c
                if keep.any():
                    sums[c, wi] += np.einsum('nc,n->c', v[keep], ew[keep], optimize=True)
    nrow = np.array([(cond == c).sum() for c in (0, 1)])
    assert (nrow > 100).all(), (subject, book, nrow)
    avg = sums / nrow[:, None, None]
    diff = avg[1] - avg[0]
    report = {'subject': subject, 'book': book, 'n_low_rows': int(nrow[0]),
              'n_high_rows': int(nrow[1]), 'n_epochs': len(x),
              'cache_bytes': xp.stat().st_size, 'meta_bytes': mp.stat().st_size,
              'contrast_rms_uv': np.sqrt((diff**2).mean(axis=1)).round(5).tolist()}
    return diff, report


def main() -> None:
    labels = pd.read_csv(LABELS, usecols=['book', 'run', 'row_idx', 'surprisal'])
    assert not labels.duplicated(['book', 'run', 'row_idx']).any()
    cuts = thresholds(labels)
    by_subject = []
    reports = []
    for subject in SUBJECTS:
        pair = []
        for book in BOOKS:
            diff, rep = one_recording(subject, book, labels, cuts[book])
            pair.append(diff)
            reports.append(rep)
            print(subject, book, rep['n_low_rows'], rep['n_high_rows'], flush=True)
        by_subject.append(np.mean(pair, axis=0))
    group = np.mean(by_subject, axis=0)
    np.savez_compressed(ROOT / 'analysis/real_topomap_arrays.npz',
                        group_mean_uv=group, subject_mean_uv=np.stack(by_subject),
                        subjects=np.asarray(SUBJECTS))
    electrode = pd.read_csv(ELECTRODES, sep='\t', encoding='utf-8-sig')
    assert electrode['name'].tolist() == [f'E{i}' for i in range(1, 129)]
    montage = mne.channels.make_dig_montage(
        ch_pos={r['name']: np.array([r['x'], r['y'], r['z']], dtype=float)
                for _, r in electrode.iterrows()}, coord_frame='head')
    info = mne.create_info(electrode['name'].tolist(), SFREQ, ch_types='eeg')
    info.set_montage(montage)
    limit = float(np.ceil(np.max(np.abs(group)) * 10) / 10)
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15), layout='constrained')
    last = None
    for ax, ((lo, hi), values) in zip(axes, zip(WINDOWS_MS, group)):
        last, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False,
                                        cmap='RdBu_r', vlim=(-limit, limit),
                                        contours=0, sensors=True,
                                        sphere=(0.0, 0.0, 0.0, 0.12), extrapolate='head')
        ax.set_title(f'{lo}–{hi} ms', fontsize=11, pad=12)
    cb = fig.colorbar(last, ax=axes, shrink=0.72, pad=0.015)
    cb.set_label('High − low surprisal (µV)', fontsize=10)
    fig.suptitle('Measured row-surprisal EEG contrast', fontsize=12, weight='bold')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    audit = {'endpoint': 'frozen row-level language-model surprisal',
             'contrast': 'high minus low within-book tertiles; exactly 10 character slots per row',
             'aggregation': 'equal mean across character epochs within row, rows within subject-book, books within subject, then 9 subjects',
             'baseline_ms': [-100, 0], 'windows_ms': WINDOWS_MS,
             'source_cache': str(EPOCHS), 'cache_audit_sha256': digest(EPOCHS / 'CACHE_AUDIT.json'),
             'label_path': str(LABELS), 'label_sha256': digest(LABELS),
             'electrodes_path': str(ELECTRODES), 'electrodes_sha256': digest(ELECTRODES),
             'subject_count': len(SUBJECTS), 'thresholds_by_book': cuts,
             'color_limit_uv': limit, 'group_rms_uv': np.sqrt((group**2).mean(axis=1)).round(5).tolist(),
             'recordings': reports,
             'interpretation': 'descriptive sensor voltage; unadjusted; not MM_mid attribution or causal surprisal response'}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print('OUTPUT', OUTPUT, 'LIMIT', limit, 'RMS', audit['group_rms_uv'])


if __name__ == '__main__':
    main()
