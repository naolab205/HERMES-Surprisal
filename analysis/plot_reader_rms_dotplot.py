#!/usr/bin/env python3
"""Show each measured reader's scalp-contrast magnitude in four time windows."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/measured_subject_window_rms_uv.csv"
OUTPUT = ROOT / "outputs/paper_figures/fig14_measured_reader_rms_dotplot.pdf"
WINDOWS = ["0-100", "100-200", "200-300", "300-400"]


def main() -> None:
    frame = pd.read_csv(SOURCE)
    assert len(frame) == 9 and frame.subject.nunique() == 9
    assert set(WINDOWS).issubset(frame.columns)
    values = frame[WINDOWS].to_numpy(dtype=float)
    assert values.shape == (9, 4) and np.isfinite(values).all() and (values >= 0).all()
    # The row order is fixed by the average of the four measured magnitudes.
    order = np.argsort(values.mean(axis=1))[::-1]
    frame = frame.iloc[order].reset_index(drop=True)
    values = frame[WINDOWS].to_numpy(dtype=float)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "figure.dpi": 180})
    fig, ax = plt.subplots(figsize=(7.2, 3.65), constrained_layout=True)
    colors = ["#477eaa", "#cb7b3e", "#4e9a85", "#8869a5"]
    offsets = [-.24, -.08, .08, .24]
    y = np.arange(len(frame))
    for column, (window, color, offset) in enumerate(zip(WINDOWS, colors, offsets)):
        ax.scatter(values[:, column], y + offset, s=43, color=color,
                   edgecolor="white", linewidth=.55, zorder=3,
                   label=f"{window} ms")
    ax.set_yticks(y, frame.subject.tolist())
    ax.invert_yaxis()
    ax.set_xlim(0, .28)
    ax.set_xlabel("Sensor RMS of high − low contrast (µV)")
    ax.set_ylabel("Reader (sorted by four-window mean)")
    ax.grid(axis="x", color="#dce3e8", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(title="Time after character onset", loc="upper center",
              bbox_to_anchor=(.5, 1.18), ncol=4, frameon=False,
              fontsize=8, title_fontsize=8)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)
    print("order:", ", ".join(frame.subject))
    print("median µV:", ", ".join(f"{v:.3f}" for v in np.median(values, axis=0)))


if __name__ == "__main__":
    main()
