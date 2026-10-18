#!/usr/bin/env python3
"""Plot fold distributions and paired ablation differences from archived CSVs.

All points are observed item-block folds. This script does not infer a sampling
distribution over readers or provide confirmatory significance tests: the same
folds were used during model development.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "outputs/paper_figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "pdf.fonttype": 42, "figure.dpi": 180,
})


def fold_values(arm: str) -> pd.DataFrame:
    frame = pd.read_csv(TABLES / f"charbag_{arm}_per_fold.csv").sort_values("fold")
    if frame.fold.tolist() != list(range(7)):
        raise ValueError(f"Incomplete seven-fold table: {arm}")
    if not all(np.isfinite(frame[c]).all() for c in
               ("trial_spearman", "item_spearman", "item_auc_hi_vs_lo")):
        raise ValueError(f"Non-finite score: {arm}")
    return frame


def boxplots() -> None:
    arms = [
        ("BL_atcnet", "ATCNet", "#84909c"),
        ("BL_conformer", "Conformer", "#a5acb5"),
        ("ABL_nochar", "No cross-char", "#d68947"),
        ("ABL_noeye", "EEG only", "#3875b9"),
        ("MM_mid", "EEG + Eye", "#8254b4"),
    ]
    metrics = [
        ("item_spearman", "Item Spearman $\\rho$", (0.27, 0.45)),
        ("item_auc_hi_vs_lo", "High--low item AUC", (0.67, 0.79)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    jitter = np.array([-0.10, -0.065, -0.03, 0.0, 0.03, 0.065, 0.10])
    for ax, (metric, ylabel, limits) in zip(axes, metrics):
        for pos, (arm, label, color) in enumerate(arms, start=1):
            values = fold_values(arm)[metric].to_numpy()
            bp = ax.boxplot(values, positions=[pos], widths=0.53, patch_artist=True,
                            showfliers=False, whis=(0, 100),
                            medianprops={"color": "#1a2430", "linewidth": 1.4},
                            boxprops={"edgecolor": "#394451", "linewidth": 0.8},
                            whiskerprops={"color": "#394451", "linewidth": 0.8},
                            capprops={"color": "#394451", "linewidth": 0.8})
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.58)
            ax.scatter(pos + jitter, values, s=13, color="#233140", zorder=3)
        ax.set_xticks(range(1, len(arms) + 1), [x[1] for x in arms],
                      rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#dce1e6", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.savefig(FIGURES / "fig05_archived_fold_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)


def heatmap() -> None:
    main = fold_values("MM_mid")["item_spearman"].to_numpy()
    arms = [
        ("ABL_noeye", "No eye"),
        ("ABL_nochar", "No cross-char"),
        ("ABL_nocsd", "No Laplacian"),
        ("ABL_norope", "No rotary"),
        ("ABL_win900", "EEG to +900 ms"),
        ("ABL_win450", "EEG to +450 ms"),
        ("MM_late", "Late fusion"),
    ]
    values = np.stack([main - fold_values(arm)["item_spearman"].to_numpy()
                       for arm, _ in arms])
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    limit = 0.06
    image = ax.imshow(values, cmap="PuOr_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                      aspect="auto")
    ax.set_xticks(range(7), [f"F{k}" for k in range(7)])
    ax.set_yticks(range(len(arms)), [label for _, label in arms])
    ax.set_xlabel("Item-block fold")
    ax.set_title("Main model minus variant: item Spearman $\\Delta\\rho$")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            val = values[row, col]
            ax.text(col, row, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7.1, color="white" if abs(val) >= 0.036 else "#1a2430")
    cb = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("Main $-$ variant, item $\\rho$")
    fig.savefig(FIGURES / "fig06_paired_ablation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def summary_csv() -> None:
    rows = []
    main = fold_values("MM_mid")
    for path in sorted(TABLES.glob("charbag_*_per_fold.csv")):
        arm = path.name.removeprefix("charbag_").removesuffix("_per_fold.csv")
        frame = fold_values(arm)
        diff = main.item_spearman.to_numpy() - frame.item_spearman.to_numpy()
        rows.append({
            "arm": arm,
            "trial_rho_mean": frame.trial_spearman.mean(),
            "trial_rho_sd": frame.trial_spearman.std(ddof=1),
            "item_rho_mean": frame.item_spearman.mean(),
            "item_rho_sd": frame.item_spearman.std(ddof=1),
            "auc_mean": frame.item_auc_hi_vs_lo.mean(),
            "auc_sd": frame.item_auc_hi_vs_lo.std(ddof=1),
            "main_minus_arm_item_rho_mean": diff.mean(),
            "main_higher_folds": int((diff > 0).sum()),
        })
    pd.DataFrame(rows).to_csv(ROOT / "analysis/extended_results_summary.csv", index=False)


if __name__ == "__main__":
    boxplots()
    heatmap()
    summary_csv()
