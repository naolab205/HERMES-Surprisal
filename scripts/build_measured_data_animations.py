"""Render a recorded EEG replay and a threshold sweep of archived predictions.

The first display uses a saved ten-second excerpt of the public, author-filtered
EEG. The second uses the existing out-of-fold prediction CSV without refitting.
Neither animation depicts model activations or an online decoding experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import spectrogram
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
DEFAULT_OUT = ROOT / 'docs/measured_demos'
BG, FG, MUTED = '#0E1B2D', '#EAF1FA', '#9EADC2'
TEAL, ORANGE = '#49CFC4', '#F4B361'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': FG, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': BG, 'figure.facecolor': BG,
                     'axes.edgecolor': '#415065'})


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract_eeg(data_root: Path, out: Path) -> None:
    from cogalign.io.events import pair_rows_rowe
    from scripts.build.build_epoch_cache import _read_raw_brainvision_robust

    reference = json.loads((ROOT / 'docs/appendix/extended/extended_extraction_provenance.json').read_text())
    # A deterministic example: first audited training recording, then its first
    # prefix-checked row. No signal amplitude or prediction selects this excerpt.
    record = reference['sources'][0]
    relative = Path(record['source_file']).relative_to('/data/my/ds004952')
    source = data_root / relative
    events = source.with_name(source.name.replace('_eeg.vhdr', '_events.tsv'))
    assert sha(source) == record['vhdr_sha256']
    assert sha(source.with_suffix('.eeg')) == record['eeg_sha256']
    assert sha(events) == record['events_sha256']
    event_frame = pd.read_csv(events, sep='\t', encoding='utf-8-sig')
    pairs = pair_rows_rowe(event_frame)
    row_idx = int(record['prefix_checked_row_indices'][0])
    onset = float(pairs.set_index('row_idx').loc[row_idx, 'start'])
    raw = _read_raw_brainvision_robust(source)
    sfreq = float(raw.info['sfreq'])
    channels = ['E11', 'E24', 'E36', 'E52', 'E62', 'E75', 'E92', 'E124']
    start = round(onset * sfreq)
    stop = start + round(10 * sfreq)
    assert stop <= raw.n_times and sfreq == 256
    values = raw.get_data(picks=channels, start=start, stop=stop) * 1e6
    assert values.shape == (8, 2560) and np.isfinite(values).all()
    selected = pairs[(pairs.start >= start / sfreq) & (pairs.start < stop / sfreq)]
    row_starts = selected.start.to_numpy() - start / sfreq
    np.savez_compressed(out / 'continuous_eeg_excerpt.npz', eeg_uv=values,
                        time_s=np.arange(values.shape[1]) / sfreq,
                        channels=np.array(channels), sfreq_hz=sfreq,
                        row_starts_s=row_starts,
                        row_indices=selected.row_idx.to_numpy())
    provenance = {
        'dataset': 'ChineseEEG / OpenNeuro ds004952',
        'source_relative_to_dataset': str(relative),
        'subject': record['subject'], 'book': record['book'], 'run': record['run'],
        'start_sample': start, 'stop_sample_exclusive': stop,
        'source_rate_hz': sfreq, 'source_start_seconds': start / sfreq,
        'duration_seconds': 10, 'first_row_index': row_idx,
        'selection': 'First audited training recording and first prefix-checked row; no signal- or score-based selection.',
        'scope': 'Continuous excerpt spans several displayed rows, not one character or one target row.',
        'source_processing': 'Dataset-author filtered_0.5_30 derivative; no additional filtering, clipping or resampling.',
        'display_processing': 'Subtract each channel median over this excerpt; use a shared microvolt scale and vertical offsets.',
        'spectral_channel': 'E62', 'channels': channels,
        'source_vhdr_sha256': sha(source),
        'source_eeg_sha256': sha(source.with_suffix('.eeg')),
        'source_events_sha256': sha(events),
        'excerpt_sha256': sha(out / 'continuous_eeg_excerpt.npz'),
    }
    (out / 'extraction_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def capture(fig) -> Image.Image:
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def save_gif(frames: list[Image.Image], target: Path, duration: int) -> dict:
    # A shared palette prevents colors and type changing between frames.
    sample = Image.new('RGB', (frames[0].width, frames[0].height * 4))
    for i, j in enumerate(np.linspace(0, len(frames) - 1, 4, dtype=int)):
        sample.paste(frames[j], (0, i * frames[0].height))
    palette = sample.quantize(colors=192)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [duration] * len(frames)
    durations[-1] = 800
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    with Image.open(target) as image:
        assert image.n_frames == len(frames)
        saved_durations = []
        for i in range(image.n_frames):
            image.seek(i)
            saved_durations.append(image.info['duration'])
        assert saved_durations == durations
    print(f'{target.name}: {len(frames)} frames, {target.stat().st_size} bytes', flush=True)
    return {'n_frames': len(frames), 'frame_ms': duration,
            'final_frame_ms': 800, 'loop_duration_ms': sum(durations),
            'sha256': sha(target)}


def style(ax) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.12)
    ax.set_axisbelow(True)


def eeg_movie(source: Path, out: Path) -> dict:
    a = np.load(source, allow_pickle=False)
    signal = a['eeg_uv'].copy()
    signal -= np.median(signal, axis=1, keepdims=True)
    time = a['time_s']
    sfreq = float(a['sfreq_hz'])
    # Shared spacing retains amplitude differences between measured channels.
    spacing = float(max(15, np.max(np.ptp(signal, axis=1)) * 1.12))
    offsets = np.arange(7, -1, -1) * spacing
    freq, spec_time, power = spectrogram(
        signal[4], fs=sfreq, window='hann', nperseg=128, noverlap=112,
        detrend='constant', scaling='density', mode='psd')
    keep = (freq >= .5) & (freq <= 30)
    db = 10 * np.log10(np.maximum(power[keep], np.finfo(float).tiny))
    low, high = np.percentile(db, [2, 98])
    fig = plt.figure(figsize=(11.2, 6.3), dpi=100)
    fig.text(.065, .945, 'Recorded EEG in motion', size=22, weight='bold')
    fig.text(.065, .897, 'ChineseEEG  /  sub-04  /  GarnettDream  /  run 02', color=MUTED, size=11)
    clock = fig.text(.94, .945, '', color=TEAL, ha='right', size=17, weight='bold')
    wave = fig.add_axes([.075, .40, .85, .425])
    traces = []
    for channel, offset in zip(signal, offsets):
        wave.plot(time, channel + offset, color='#365168', lw=.65)
        line, = wave.plot([], [], color=TEAL, lw=1.3)
        traces.append(line)
    wave.set(xlim=(0, 10), ylim=(-spacing * .65, spacing * 7.65),
             yticks=offsets, yticklabels=a['channels'].tolist(), xticks=[0, 2, 4, 6, 8, 10])
    wave.tick_params(axis='y', length=0)
    wave.spines['left'].set_visible(False)
    wave.set_title('8 measured channels · shared amplitude scale', loc='left', color=FG, pad=10, size=11)
    for onset in a['row_starts_s']:
        wave.axvline(onset, color=ORANGE, alpha=.5, lw=.7, ls='--')
    wave.plot([9.55, 9.55], [offsets[-1], offsets[-1] + 10], color=FG, lw=2)
    wave.text(9.35, offsets[-1] + 5, '10 µV', ha='right', va='center', size=9)
    cursor = wave.axvline(0, color='#FFFFFF', lw=.9, alpha=.9)
    style(wave)
    spec = fig.add_axes([.075, .155, .80, .155])
    im = spec.pcolormesh(spec_time, freq[keep], db, shading='nearest',
                         cmap='magma', vmin=low, vmax=high)
    spec.set(xlim=(0, 10), ylim=(.5, 30), ylabel='Hz', xlabel='Seconds from excerpt start', yticks=[5, 15, 25])
    spec.set_title('E62 · measured spectrogram', loc='left', size=10, color=FG, pad=7)
    cax = fig.add_axes([.89, .155, .014, .155])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('dB µV²/Hz', size=8, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    scursor = spec.axvline(0, color='#FFFFFF', lw=1.1)
    fig.text(.065, .035, '256 Hz · 0.5–30 Hz author-filtered EEG · dashed lines: observed row starts · offline replay', size=9, color=MUTED)
    frames = []
    for now in np.linspace(0, time[-1], 128):
        active = (time <= now) & (time >= now - .8)
        for line, channel, offset in zip(traces, signal, offsets):
            line.set_data(time[active], channel[active] + offset)
        cursor.set_xdata([now, now])
        scursor.set_xdata([now, now])
        clock.set_text(f'{now:04.1f} / 10.0 s')
        frames.append(capture(fig))
    frames[64].save(out / 'continuous_eeg_still.png')
    plt.close(fig)
    result = save_gif(frames, out / 'continuous_eeg.gif', 80)
    result.update({'source_sha256': sha(source), 'sample_rate_hz': sfreq,
                   'signal_duration_s': 10, 'signal_resampling': False,
                   'per_channel_normalization': False,
                   'shared_channel_spacing_uv': spacing,
                   'spectrogram': {'channel': 'E62', 'nperseg': 128, 'noverlap': 112,
                                   'color_limits_db': [float(low), float(high)],
                                   'note': 'Offline whole-excerpt spectrum; color limits are display-only percentiles.'}})
    return result


def roc_movie(predictions: Path, out: Path) -> dict:
    data = pd.read_csv(predictions)
    assert data.groupby(['fold', 'item_id']).tercile.nunique().max() == 1
    item = data.groupby(['fold', 'item_id'], sort=True).agg(
        pred=('pred', 'mean'), tercile=('tercile', 'first')).reset_index()
    curves = []
    colors = [TEAL, '#F4B361', '#A89AF2', '#71A8F4', '#F190AB', '#B3D576', '#E2C3A0']
    for fold, part in item[item.tercile != 1].groupby('fold', sort=True):
        y = (part.tercile.to_numpy() == 2).astype(int)
        scores = part.pred.to_numpy()
        fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
        curves.append({'fold': int(fold), 'y': y, 'scores': scores,
                       'fpr': fpr, 'tpr': tpr, 'auc': float(roc_auc_score(y, scores))})
    assert len(curves) == 7
    expected = pd.read_csv(ROOT / 'results/tables/charbag_MM_mid_per_fold.csv').sort_values('fold')
    assert np.allclose([c['auc'] for c in curves], expected.item_auc_hi_vs_lo, atol=1e-12, rtol=0)
    fig = plt.figure(figsize=(11.2, 5.5), dpi=100)
    fig.text(.065, .93, 'What the AUC represents', size=22, weight='bold')
    fig.text(.065, .875, 'Saved item predictions · high vs low surprisal · threshold sweep, without refitting', size=10, color=MUTED)
    ax = fig.add_axes([.075, .19, .36, .57])
    ax.plot([0, 1], [0, 1], '--', lw=.9, color='#677B91', alpha=.7)
    markers = []
    for c, color in zip(curves, colors):
        ax.step(c['fpr'], c['tpr'], where='post', lw=1.5, color=color,
                label=f"{c['fold']}  {c['auc']:.3f}", alpha=.85)
        dot, = ax.plot([], [], 'o', ms=6, color=color, markeredgecolor=BG)
        markers.append(dot)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate', ylabel='True positive rate')
    ax.set_title('Seven held-out item blocks', loc='left', color=FG, size=11)
    legend = ax.legend(title='Block / AUC', loc='lower right', ncol=2, fontsize=8,
                       title_fontsize=8, frameon=False, labelcolor=FG)
    legend.get_title().set_color(MUTED)
    style(ax)
    hist = fig.add_axes([.54, .36, .395, .40])
    primary = curves[0]
    scores, y = primary['scores'], primary['y']
    bins = np.linspace(scores.min(), scores.max(), 32)
    hist.hist(scores[y == 0], bins=bins, color='#71A8F4', alpha=.7, label='Low surprisal')
    hist.hist(scores[y == 1], bins=bins, color=ORANGE, alpha=.7, label='High surprisal')
    threshold_line = hist.axvline(scores.max(), color=TEAL, lw=2)
    hist.set(xlabel='Item-mean predicted score (training SD units)', ylabel='Items')
    hist.set_title('Block 0 · moving score threshold', loc='left', size=11, color=FG)
    hist.legend(frameon=False, fontsize=8, labelcolor=FG)
    style(hist)
    tally = fig.text(.54, .245, '', size=12, color=TEAL)
    counts = fig.text(.54, .193, '', size=10, color=MUTED)
    label = fig.text(.935, .93, '', ha='right', size=11, color=TEAL)
    mean_auc = float(np.mean([c['auc'] for c in curves]))
    fig.text(.065, .073, f'Archived mean AUC {mean_auc:.3f} · separate threshold quantiles within each block · no threshold selected', size=9, color=MUTED)
    fig.text(.065, .035, 'Readers recur across blocks; this is a retrospective view of saved predictions, not an online or new-reader test.', size=9, color=MUTED)
    frames, records = [], []
    for frame, fraction in enumerate(np.linspace(0, 1, 100)):
        for c, dot in zip(curves, markers):
            threshold = float(np.quantile(c['scores'], 1 - fraction))
            if frame == 0:
                threshold = float(np.nextafter(c['scores'].max(), np.inf))
            elif frame == 99:
                threshold = float(np.nextafter(c['scores'].min(), -np.inf))
            called_high = c['scores'] >= threshold
            truth = c['y'].astype(bool)
            tp, fp = int(np.sum(called_high & truth)), int(np.sum(called_high & ~truth))
            fn, tn = int(np.sum(~called_high & truth)), int(np.sum(~called_high & ~truth))
            tpr, fpr = tp / (tp + fn), fp / (fp + tn)
            dot.set_data([fpr], [tpr])
            records.append({'frame': frame, 'fold': c['fold'], 'threshold': threshold,
                            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr})
            if c['fold'] == 0:
                threshold_line.set_xdata([threshold, threshold])
                tally.set_text(f'TPR  {tpr:.1%}       FPR  {fpr:.1%}')
                counts.set_text(f'TP {tp}   FP {fp}   TN {tn}   FN {fn}    |    threshold {threshold:.3f}')
        label.set_text(f'Threshold sweep  {fraction:.0%}')
        frames.append(capture(fig))
    frames[50].save(out / 'prediction_threshold_still.png')
    plt.close(fig)
    pd.DataFrame(records).to_csv(out / 'threshold_frame_metrics.csv', index=False)
    result = save_gif(frames, out / 'prediction_threshold.gif', 100)
    result.update({'source_sha256': sha(predictions), 'source': str(predictions.relative_to(ROOT)),
                   'n_unique_items_all_terciles': len(item),
                   'n_items_outer_terciles': sum(len(c['y']) for c in curves),
                   'fold_auc': [c['auc'] for c in curves], 'mean_fold_auc': mean_auc,
                   'frame_metrics_sha256': sha(out / 'threshold_frame_metrics.csv'),
                   'auc_matches_original_fold_table': True,
                   'threshold_selection': 'Display-only score quantiles computed separately within each fold; no operating threshold selected.',
                   'time_axis': False, 'new_model_fitting': False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_OUT / 'continuous_eeg_excerpt.npz')
    parser.add_argument('--extract-eeg', action='store_true')
    parser.add_argument('--data-root', type=Path, help='Public ChineseEEG dataset root; required only for --extract-eeg')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_eeg:
        if args.data_root is None:
            parser.error('--extract-eeg requires --data-root')
        extract_eeg(args.data_root, args.output_dir)
        args.source = args.output_dir / 'continuous_eeg_excerpt.npz'
    eeg = eeg_movie(args.source, args.output_dir)
    roc = roc_movie(ROOT / 'results/tables/charbag_MM_mid_predictions.csv', args.output_dir)
    (args.output_dir / 'rendering_provenance.json').write_text(json.dumps(
        {'continuous_eeg': eeg, 'prediction_threshold': roc,
         'synthetic_signals': False, 'model_activation_visualization': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
