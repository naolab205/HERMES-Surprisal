#!/usr/bin/env python3
"""Describe held-out item correlations within each book and in the pooled set.

The archived predictions already reflect model development on these seven blocks.
These plots are descriptive and are not independent validation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/tables/charbag_MM_mid_predictions.csv"
OUT = ROOT / "outputs/paper_figures/fig11_book_stratified_forest.pdf"


def main() -> None:
    df = pd.read_csv(PREDICTIONS)
    assert len(df) == 102499 and set(df.fold) == set(range(7))
    rows = []
    for fold, held_out in df.groupby("fold"):
        for book, group in [("Pooled", held_out), *held_out.groupby("book")]:
            items = group.groupby("item_id", as_index=False).agg(
                y=("y", "first"), pred=("pred", "mean"))
            rows.append({"book": book, "fold": fold, "items": len(items),
                         "item_spearman": spearmanr(items.y, items.pred).statistic})
    result = pd.DataFrame(rows).sort_values(["book", "fold"])
    result.to_csv(ROOT / "analysis/book_stratified_item_correlations.csv", index=False)

    order = ["Pooled", "GarnettDream", "LittlePrince"]
    labels = ["Both books (pooled)", "Garnett Dream", "Little Prince"]
    colors = ["#5c668a", "#356f9e", "#b5743c"]
    fig, ax = plt.subplots(figsize=(7.0, 2.65), constrained_layout=True)
    jitter = np.array([-.11, -.075, -.04, 0, .04, .075, .11])
    for pos, (book, color) in enumerate(zip(order, colors), start=1):
        values = result[result.book == book].sort_values("fold").item_spearman.to_numpy()
        mean, sd = values.mean(), values.std(ddof=1)
        ax.axhspan(pos - .27, pos + .27, color=color, alpha=.045, zorder=0)
        ax.scatter(values, pos + jitter, s=23, facecolor='white',
                   edgecolor=color, linewidth=.9, zorder=3)
        ax.errorbar(mean, pos, xerr=sd, fmt='D', color=color, ecolor=color,
                    markersize=7, capsize=5, elinewidth=2.4, zorder=4)
        ax.text(.485, pos, f'{mean:.3f} ± {sd:.3f}', va='center', ha='left',
                color='#24313b', fontsize=8.6)
    ax.set_yticks(range(1, 4), labels)
    ax.invert_yaxis()
    ax.set_xlim(.20, .59)
    ax.set_xticks([.20, .25, .30, .35, .40, .45])
    ax.set_xlabel('Held-out item Spearman $\\rho$')
    ax.grid(axis='x', color='#e0e5e9', lw=.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(result.groupby("book").item_spearman.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
