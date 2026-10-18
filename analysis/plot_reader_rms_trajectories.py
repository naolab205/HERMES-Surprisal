#!/usr/bin/env python3
"""Plot paired reader trajectories of measured sensor-RMS contrasts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/measured_subject_window_rms_uv.csv"
OUTPUT = ROOT / "outputs/paper_figures/fig13_measured_reader_rms_trajectories.pdf"
WINDOWS = ["0-100", "100-200", "200-300", "300-400"]


def main() -> None:
    frame = pd.read_csv(SOURCE)
    assert len(frame) == 9 and frame.subject.nunique() == 9
    values = frame[WINDOWS].to_numpy(dtype=float)
    assert values.shape == (9, 4) and np.isfinite(values).all()
    assert (values >= 0).all()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "figure.dpi": 180})
    fig, ax = plt.subplots(figsize=(7.2, 3.25), constrained_layout=True)
    x = np.arange(len(WINDOWS))
    for row in values:
        ax.plot(x, row, color="#7194a9", lw=1.15, alpha=.65,
                marker="o", markersize=3.5, zorder=2)
    median = np.median(values, axis=0)
    ax.plot(x, median, color="#b35932", lw=2.8, marker="D",
            markersize=6, markeredgecolor="white", markeredgewidth=.6,
            label="Across-reader median", zorder=4)
    ax.plot([], [], color="#7194a9", lw=1.15, marker="o",
            markersize=3.5, label="Individual reader (n = 9)")

    ax.set_xticks(x, ["0–100", "100–200", "200–300", "300–400"])
    ax.set_xlabel("Time after character onset (ms)")
    ax.set_ylabel("Sensor RMS of high − low contrast (µV)")
    ax.set_xlim(-.12, 3.12)
    ax.set_ylim(0, .29)
    ax.grid(axis="y", color="#dfe4e8", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)
    print("median µV:", ", ".join(f"{v:.3f}" for v in median))


if __name__ == "__main__":
    main()
