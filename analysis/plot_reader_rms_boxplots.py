#!/usr/bin/env python3
"""Boxplots of measured reader-level scalp contrast magnitude by time window."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "analysis/measured_subject_window_rms_uv.csv"
output = ROOT / "outputs/paper_figures/fig12_measured_reader_rms_boxplots.pdf"
frame = pd.read_csv(source)
windows = ["0-100", "100-200", "200-300", "300-400"]
assert len(frame) == 9 and frame.subject.nunique() == 9
values = [frame[window].to_numpy(dtype=float) for window in windows]
assert all(np.isfinite(v).all() and (v >= 0).all() for v in values)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "figure.dpi": 180})
fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
colors = ["#88a8bb", "#5085a6", "#c17f50", "#a9a3b8"]
bp = ax.boxplot(values, positions=np.arange(1, 5), widths=.55,
                whis=(0, 100), showfliers=False, patch_artist=True,
                medianprops={"color": "#172b3a", "linewidth": 1.8},
                boxprops={"edgecolor": "#344756", "linewidth": .85},
                whiskerprops={"color": "#344756", "linewidth": .85},
                capprops={"color": "#344756", "linewidth": .85})
for box, color in zip(bp["boxes"], colors):
    box.set_facecolor(color)
    box.set_alpha(.67)
offset = np.linspace(-.14, .14, len(frame))
for pos, v in enumerate(values, start=1):
    ax.scatter(pos + offset, v, s=24, facecolor="#202d36",
               edgecolor="white", linewidth=.5, zorder=4)
    median = np.median(v)
    ax.text(pos, .279, f"median {median:.3f}", ha="center", va="top",
            fontsize=8, color="#263946")
ax.set_xticks(range(1, 5), [f"{window} ms" for window in windows])
ax.set_xlabel("Time after character onset")
ax.set_ylabel("Reader-level sensor RMS of high − low contrast (µV)")
ax.set_ylim(0, .29)
ax.grid(axis="y", color="#dde3e7", linewidth=.7)
ax.set_axisbelow(True)
fig.savefig(output, bbox_inches="tight")
print(output)
print("medians", [round(float(np.median(v)), 3) for v in values])
