#!/usr/bin/env python3
"""Descriptive Ward dendrogram with original reader-level EEG contrasts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/measured_subject_window_rms_uv.csv"
OUTPUT = ROOT / "outputs/paper_figures/fig15_measured_reader_rms_clustering.pdf"
WINDOWS = ["0-100", "100-200", "200-300", "300-400"]


def main() -> None:
    frame = pd.read_csv(SOURCE)
    assert len(frame) == 9 and frame.subject.nunique() == 9
    original = frame[WINDOWS].to_numpy(dtype=float)
    assert original.shape == (9, 4) and np.isfinite(original).all()
    assert (original >= 0).all()
    scaled = (original - original.mean(axis=0)) / original.std(axis=0, ddof=1)
    tree = linkage(scaled, method="ward", optimal_ordering=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "figure.dpi": 180})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.8),
                                     gridspec_kw={"width_ratios": [1.05, 1.45]},
                                     constrained_layout=True)
    drawn = dendrogram(tree, orientation="left", labels=frame.subject.tolist(),
                       ax=left, color_threshold=0,
                       above_threshold_color="#526677")
    leaves = drawn["leaves"]
    y = 5 + 10 * np.arange(len(leaves))
    colors = ["#477eaa", "#cb7b3e", "#4e9a85", "#8869a5"]
    offsets = [-2.5, -.8, .8, 2.5]
    for col, (window, color, offset) in enumerate(zip(WINDOWS, colors, offsets)):
        right.scatter(original[leaves, col], y + offset, s=34, color=color,
                      edgecolor="white", linewidth=.45, zorder=3,
                      label=f"{window} ms")
    right.set_ylim(left.get_ylim())
    right.set_yticks(y, frame.subject.iloc[leaves].tolist())
    right.yaxis.tick_right()
    right.tick_params(axis="y", length=0)
    right.set_xlim(0, .28)
    right.set_xlabel("Sensor RMS of high − low contrast (µV)")
    right.grid(axis="x", color="#dce3e8", linewidth=.65)
    right.set_axisbelow(True)
    right.legend(title="Time after character onset", loc="upper center",
                 bbox_to_anchor=(.5, 1.23), ncol=2, frameon=False,
                 fontsize=7.5, title_fontsize=7.5)
    left.set_xlabel("Ward merge distance (standardized inputs)")
    left.set_yticks([])
    left.spines[["top", "right"]].set_visible(False)
    right.spines[["top", "left"]].set_visible(False)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)
    print("leaf order:", [frame.subject.iloc[i] for i in leaves])


if __name__ == "__main__":
    main()
