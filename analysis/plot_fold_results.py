"""Render source-consistent vector figures from the archived fold tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "model_code/analysis" / "tables"
OUT = ROOT / "els-cas-templates" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.labelsize": 9.5, "axes.titlesize": 10.5,
                     "pdf.fonttype": 42, "figure.dpi": 180})


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
    fig.savefig(OUT / "fig03_exploratory_fold_results.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results()
