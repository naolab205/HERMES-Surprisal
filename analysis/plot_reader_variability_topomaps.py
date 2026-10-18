#!/usr/bin/env python3
"""Map between-reader SD of the measured high-minus-low EEG contrast.

Input arrays come from plot_real_surprisal_topomaps.py: each reader contributes
equally, after averaging the two books. This is a descriptive map, not a
significance test or a model-attribution map.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / 'analysis/real_topomap_arrays.npz'
ELECTRODES = Path('/data/my/ds004952/sub-04/ses-LittlePrince/eeg/sub-04_ses-LittlePrince_space-CapTrak_electrodes.tsv')
OUT = ROOT / 'outputs/paper_figures/fig12_measured_reader_variability_topomaps.pdf'
WINDOWS = ('0–100 ms', '100–200 ms', '200–300 ms', '300–400 ms')


def main() -> None:
    saved = np.load(ARRAYS)
    subject = saved['subject_mean_uv']
    assert subject.shape == (9, 4, 128) and np.isfinite(subject).all()
    sd = subject.std(axis=0, ddof=1)
    assert sd.shape == (4, 128) and (sd >= 0).all()
    electrode = pd.read_csv(ELECTRODES, sep='\t', encoding='utf-8-sig')
    assert electrode['name'].tolist() == [f'E{i}' for i in range(1, 129)]
    montage = mne.channels.make_dig_montage(
        ch_pos={r['name']: np.array([r['x'], r['y'], r['z']], dtype=float)
                for _, r in electrode.iterrows()}, coord_frame='head')
    info = mne.create_info(electrode['name'].tolist(), 128.0, ch_types='eeg')
    info.set_montage(montage)
    limit = float(np.ceil(sd.max() * 10) / 10)
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.05), layout='constrained')
    image = None
    for ax, title, values in zip(axes, WINDOWS, sd):
        image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False,
                                         cmap='YlOrRd', vlim=(0, limit),
                                         contours=0, sensors=True,
                                         sphere=(0., 0., 0., .12), extrapolate='head')
        ax.set_title(title, fontsize=10, pad=10)
    cb = fig.colorbar(image, ax=axes, shrink=.72, pad=.012)
    cb.set_label('Across-reader SD (µV)', fontsize=9)
    fig.suptitle('Reader variability of the measured surprisal contrast',
                 fontsize=11, weight='bold')
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    fig.savefig(OUT.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    frame = pd.DataFrame(sd.T, columns=['0-100', '100-200', '200-300', '300-400'])
    frame.insert(0, 'electrode', electrode['name'])
    frame.to_csv(ROOT / 'analysis/measured_reader_variability_by_electrode.csv', index=False)
    audit = {
        'input_arrays': str(ARRAYS),
        'input_arrays_sha256': hashlib.sha256(ARRAYS.read_bytes()).hexdigest(),
        'electrodes': str(ELECTRODES),
        'electrodes_sha256': hashlib.sha256(ELECTRODES.read_bytes()).hexdigest(),
        'statistic': 'sample SD across nine reader-level signed high-minus-low contrasts, ddof=1',
        'windows_ms': [[0, 100], [100, 200], [200, 300], [300, 400]],
        'sensor_mean_sd_uv': sd.mean(axis=1).round(6).tolist(),
        'shared_color_limit_uv': limit,
        'interpretation': 'descriptive reader variability; not a significance or attribution map',
    }
    (ROOT / 'analysis/reader_variability_audit.json').write_text(
        json.dumps(audit, indent=2) + '\n')
    print('sensor-mean SD per window', sd.mean(axis=1).round(4).tolist(),
          'shared color scale', limit)


if __name__ == '__main__':
    main()
