#!/usr/bin/env python3
"""Make descriptive result figures from archived predictions and measured EEG.

These figures do not train or select a model. Item-level predictions are
out-of-fold, while EEG sensor contrasts are observational and unadjusted.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'outputs/paper_figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'figure.dpi': 180})


def item_distributions() -> None:
    data = pd.read_csv(ROOT / 'results/tables/charbag_MM_mid_predictions.csv')
    assert len(data) == 102499
    item = data.groupby(['book', 'fold', 'item_id'], as_index=False).agg(
        y=('y', 'first'), tercile=('tercile', 'first'), pred=('pred', 'mean'),
        readers=('subject', 'nunique'))
    assert len(item) == 11037
    item.to_csv(ROOT / 'analysis/item_prediction_distribution_source.csv', index=False)
    colors = ['#467cae', '#9b9fa6', '#cf8450']
    names = ['Low', 'Middle', 'High']
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True,
                             constrained_layout=True)
    for ax, (book, group) in zip(axes, item.groupby('book')):
        values = [group.loc[group.tercile.eq(i), 'pred'].to_numpy() for i in range(3)]
        vp = ax.violinplot(values, positions=[1, 2, 3], widths=.78,
                           showextrema=False, showmedians=False)
        for body, color in zip(vp['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor('#243341')
            body.set_alpha(.65)
        for i, v in enumerate(values, start=1):
            q1, med, q3 = np.quantile(v, [.25, .5, .75])
            ax.plot([i, i], [q1, q3], color='#182832', lw=3, zorder=3)
            ax.plot([i-.11, i+.11], [med, med], color='white', lw=1.6, zorder=4)
            ax.text(i, -1.65, f'n={len(v):,}', ha='center', fontsize=7.7)
        ax.set_title('Garnett Dream' if book == 'GarnettDream' else 'Little Prince')
        ax.set_xticks([1, 2, 3], names)
        ax.set_xlabel('Frozen row-surprisal tercile')
        ax.set_ylim(-1.8, 2.1)
        ax.grid(axis='y', color='#e0e5e9', lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Held-out item-mean prediction')
    fig.savefig(FIG / 'fig08_heldout_prediction_distributions.pdf', bbox_inches='tight')
    plt.close(fig)
    summary = item.groupby(['book', 'tercile']).pred.agg(['count', 'median', 'mean']).reset_index()
    summary.to_csv(ROOT / 'analysis/item_prediction_distribution_summary.csv', index=False)
    print(summary.to_string(index=False))


def scalp_heterogeneity() -> None:
    data = np.load(ROOT / 'analysis/real_topomap_arrays.npz')
    subject = data['subject_mean_uv']
    names = data['subjects']
    assert subject.shape == (9, 4, 128)
    rms = np.sqrt(np.mean(subject ** 2, axis=2))
    assert np.isfinite(rms).all()
    frame = pd.DataFrame(rms, columns=['0-100', '100-200', '200-300', '300-400'])
    frame.insert(0, 'subject', names)
    frame.to_csv(ROOT / 'analysis/measured_subject_window_rms_uv.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.45), constrained_layout=True)
    times = np.array([50, 150, 250, 350])
    for i, name in enumerate(names):
        color = '#c67936' if name == 'sub-05' else '#3f76a5' if name == 'sub-07' else '#a8b0b7'
        weight = 1.8 if name in ('sub-05', 'sub-07') else .9
        alpha = 1 if name in ('sub-05', 'sub-07') else .72
        ax.plot(times, rms[i], marker='o', markersize=3.5, lw=weight,
                color=color, alpha=alpha, zorder=2 if name in ('sub-05', 'sub-07') else 1,
                label=name if name in ('sub-05', 'sub-07') else None)
    median = np.median(rms, axis=0)
    ax.plot(times, median, marker='D', markersize=5, lw=2.4,
            color='#252c37', label='Reader median', zorder=4)
    ax.set_xticks(times, ['0–100', '100–200', '200–300', '300–400'])
    ax.set_xlabel('Time after character onset (ms)')
    ax.set_ylabel('Sensor RMS, high − low contrast (µV)')
    ax.set_ylim(0, .29)
    ax.grid(axis='y', color='#dfe4e8', linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper right', fontsize=8)
    fig.savefig(FIG / 'fig10_measured_reader_timecourses.pdf', bbox_inches='tight')
    plt.close(fig)
    print('RMS range', rms.min(), rms.max(), 'median', np.median(rms))


if __name__ == '__main__':
    item_distributions()
    scalp_heterogeneity()
