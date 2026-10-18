"""Render source-consistent vector figures from the archived fold tables."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.labelsize": 9.5, "axes.titlesize": 10.5,
                     "pdf.fonttype": 42, "figure.dpi": 180})


def box(ax, xy, w, h, title, subtitle, color):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.016,rounding_size=0.025",
                           facecolor=color, edgecolor="#303745", linewidth=0.9)
    ax.add_patch(patch)
    ax.text(xy[0] + w/2, xy[1] + h*0.64, title, ha="center", va="center",
            fontsize=8.4, weight="bold")
    ax.text(xy[0] + w/2, xy[1] + h*0.29, subtitle, ha="center", va="center",
            fontsize=6.9)


def arrow(ax, a, b, color="#424a59"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.2, color=color))


def architecture():
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, (.02,.65), .21,.22, "EEG epoch", "128 channels\n-100 to 1200 ms", "#dceafe")
    box(ax, (.275,.65), .20,.22, "EEG encoder", "CSD; CNN; temporal\nsequence block", "#dceafe")
    box(ax, (.02,.31), .21,.22, "Eye epoch", "gaze, pupil, valid\n-100 to 800 ms", "#dff2e6")
    box(ax, (.275,.31), .20,.22, "Eye encoder", "CNN; temporal\nsequence block", "#dff2e6")
    box(ax, (.535,.48), .18,.22, "Residual gate", "EEG + gate(Eye)\nzero initialized", "#f9e6d0")
    box(ax, (.765,.48), .21,.22, "Row prediction", "slot sequence; mean\npooling; MLP", "#eee4fa")
    arrow(ax,(.23,.76),(.275,.76)); arrow(ax,(.23,.42),(.275,.42))
    arrow(ax,(.475,.76),(.535,.61)); arrow(ax,(.475,.42),(.535,.55))
    arrow(ax,(.715,.59),(.765,.59))
    ax.text(.5,.12, "Frozen language model scores each displayed row; the score is a training label only.",
            ha="center", va="center", fontsize=8.7)
    ax.plot([.02,.98],[.21,.21], color="#999", linewidth=.7, linestyle="--")
    fig.savefig(OUT / "fig1_combined_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def folds(name):
    df = pd.read_csv(TABLES / f"charbag_{name}_per_fold.csv").sort_values("fold")
    assert df.fold.tolist() == list(range(7)), name
    return df


def results():
    old = pd.read_csv(TABLES / "surprisal_cv_full_per_fold.csv")
    old = old[old.model == "CogAlign_eeg_only"].sort_values("fold")
    sets = [("Frozen row EEG",old,"#7d8597"),
            ("Combined EEG + Eye",folds("MM_mid"),"#7e57c2"),
            ("EEG only, matched",folds("ABL_noeye"),"#2776bd"),
            ("No character block",folds("ABL_nochar"),"#d47c26")]
    metrics = [("trial_spearman","Single reading Spearman $\\rho$"),
               ("item_spearman","Item Spearman $\\rho$"),
               ("item_auc_hi_vs_lo","Item high-low AUC")]
    fig, axes = plt.subplots(1,3,figsize=(7.1,2.75), constrained_layout=True)
    rng = np.random.default_rng(1)
    for ax,(metric,title) in zip(axes,metrics):
        for j,(label,data,color) in enumerate(sets):
            vals = data[metric].to_numpy(dtype=float)
            ax.bar(j, vals.mean(), width=.65, color=color, alpha=.83)
            ax.scatter(j+rng.uniform(-.12,.12,len(vals)), vals, s=12,
                       color="#243043", zorder=3)
            ax.errorbar(j, vals.mean(), yerr=vals.std(ddof=1), fmt="none",
                        ecolor="#16202e", capsize=3, linewidth=1, zorder=4)
        ax.set_title(title)
        ax.set_xticks(range(4),["Old EEG","EEG+Eye","EEG","No char"], rotation=34, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim((.08,.43) if metric != "item_auc_hi_vs_lo" else (.62,.79))
    fig.savefig(OUT / "fig2_fold_results.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture(); results()
