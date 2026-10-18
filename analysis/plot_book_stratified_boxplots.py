#!/usr/bin/env python3
"""Boxplots of archived held-out item correlation by book, with all seven folds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/book_stratified_item_correlations.csv"
OUTPUT = ROOT / "outputs/paper_figures/fig07_book_stratified_boxplots.pdf"
GROUPS = [("Pooled", "Both books\n(pooled)"),
          ("GarnettDream", "Garnett Dream"),
          ("LittlePrince", "Little Prince")]


def main() -> None:
    frame = pd.read_csv(SOURCE)
    values = []
    for book, _ in GROUPS:
        group = frame.loc[frame.book.eq(book)].sort_values("fold")
        assert group.fold.tolist() == list(range(7))
        row = group.item_spearman.to_numpy(dtype=float)
        assert np.isfinite(row).all()
        values.append(row)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "figure.dpi": 180})
    fig, ax = plt.subplots(figsize=(6.8, 3.25), constrained_layout=True)
    colors = ["#687792", "#638daf", "#c08b62"]
    bp = ax.boxplot(values, widths=.48, whis=(0, 100), showfliers=False,
                    patch_artist=True,
                    medianprops={"color": "#182a37", "linewidth": 1.7},
                    boxprops={"edgecolor": "#3d4c59", "linewidth": .9},
                    whiskerprops={"color": "#3d4c59", "linewidth": .9},
                    capprops={"color": "#3d4c59", "linewidth": .9})
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(.62)
    jitter = np.array([-.15, -.10, -.05, 0, .05, .10, .15])
    for pos, row in enumerate(values, 1):
        ax.scatter(pos + jitter, row, s=27, facecolor="#263744",
                   edgecolor="white", linewidth=.5, zorder=4)
    ax.set_xticks(range(1, 4), [label for _, label in GROUPS])
    ax.set_ylabel("Held-out item Spearman $\\rho$")
    ax.set_ylim(.20, .46)
    ax.grid(axis="y", color="#dce2e6", linewidth=.7)
    ax.set_axisbelow(True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)
    print("means:", [round(float(v.mean()), 3) for v in values])


if __name__ == "__main__":
    main()
