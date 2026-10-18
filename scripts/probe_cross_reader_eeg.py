#!/usr/bin/env python3
"""Training-block-only probe of cross-reader row EEG repeatability.

This is a feasibility check for reader-invariant pretraining, not a surprisal
decoder evaluation. Same-item pairs are compared with nearby, same-book,
same-run, same-count different-item pairs. No target values enter the probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foundation.data import ShardedEpochs, attach_labels_and_splits, load_char_bags  # noqa: E402

OUT = ROOT / "results" / "cross_reader_probe_20261002"
SEED = 20261002
SUBJECTS = ("sub-04", "sub-05", "sub-06")
SLOTS = (0, 3, 6, 9)
POST_BINS = ((13, 39), (39, 65), (65, 90), (90, 129), (129, 167))


def features(shards: ShardedEpochs, slot_index: list[np.ndarray], obs_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    post, pre = [], []
    for start in range(0, len(obs_indices), 64):
        chunk = obs_indices[start:start + 64]
        ids = np.concatenate([slot_index[int(i)][list(SLOTS)] for i in chunk])
        x = shards.take(ids).reshape(len(chunk), len(SLOTS), 128, 167)
        base = x[..., :13].mean(axis=-1, keepdims=True)
        post.append(np.concatenate([(x[..., a:b] - base).mean(axis=(1, 3)) for a, b in POST_BINS], axis=1))
        pre.append(x[..., :13].mean(axis=(1, 3)))
    return np.concatenate(post).astype(np.float32), np.concatenate(pre).astype(np.float32)


def center_scale(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = x[train_mask].mean(axis=0)
    scale = x[train_mask].std(axis=0).clip(min=1e-3)
    return (x - mean) / scale


def negative_index(table: pd.DataFrame, eval_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = []
    for i in eval_idx:
        row = table.iloc[int(i)]
        same = table.iloc[eval_idx]
        near = eval_idx[(same["book"].to_numpy() == row.book)
                        & (same["run"].to_numpy() == row.run)
                        & (np.abs(same["row_idx"].to_numpy() - row.row_idx) >= 5)
                        & (np.abs(same["row_idx"].to_numpy() - row.row_idx) <= 60)]
        if not len(near):
            near = eval_idx[(same["book"].to_numpy() == row.book)
                            & (same["run"].to_numpy() == row.run)
                            & (same["row_idx"].to_numpy() != row.row_idx)]
        result.append(int(rng.choice(near)) if len(near) else -1)
    return np.asarray(result)


def paired_auc(a: np.ndarray, b: np.ndarray, neg: np.ndarray) -> float:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-6)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-6)
    pos_score = np.sum(a * b, axis=1)
    neg_score = np.sum(a * b[neg], axis=1)
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(a))],
                               np.r_[pos_score, neg_score]))


def matched_win(a: np.ndarray, b: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-6)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-6)
    advantage = np.sum(a * b, axis=1) - np.sum(a * b[neg], axis=1)
    wins = (advantage > 0).astype(float) + .5 * (advantage == 0)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    samples = bootstrap_rng.choice(wins, size=(2000, len(wins)), replace=True).mean(axis=1)
    return {"win_rate": float(wins.mean()),
            "bootstrap_95ci": [float(x) for x in np.quantile(samples, [.025, .975])]}


def main() -> None:
    rng = np.random.default_rng(SEED)
    bags = load_char_bags(subjects=SUBJECTS)
    frame = bags.obs.copy()
    frame["obs_index"] = np.arange(len(frame))
    frame = attach_labels_and_splits(frame, fold=0)
    frame = frame[(frame.split == "train") & (frame.n_slots == 10)
                  & (frame.n_char == 10)
                  & frame.subject.isin(SUBJECTS)]
    pivot = frame.pivot(index="item_id", columns="subject", values="obs_index").dropna()
    info = frame.drop_duplicates("item_id").set_index("item_id")
    table = info.loc[pivot.index, ["book", "run", "row_idx", "n_slots", "n_char"]].reset_index()
    pivot = pivot.reset_index(drop=True)
    # Deterministic, book-stratified cap; no label-based sampling.
    keep = []
    for book, cap in (("GarnettDream", 1000), ("LittlePrince", 600)):
        ids = np.flatnonzero(table.book.to_numpy() == book)
        keep.extend(rng.choice(ids, min(cap, len(ids)), replace=False))
    keep = np.sort(keep)
    table, pivot = table.iloc[keep].reset_index(drop=True), pivot.iloc[keep].reset_index(drop=True)
    train_mask = np.zeros(len(table), dtype=bool)
    for book in table.book.unique():
        ids = np.flatnonzero(table.book.to_numpy() == book)
        train_mask[rng.choice(ids, int(round(len(ids) * .7)), replace=False)] = True
    eval_idx = np.flatnonzero(~train_mask)
    neg = negative_index(table, eval_idx, rng)
    valid = neg >= 0
    eval_idx, neg = eval_idx[valid], neg[valid]
    # ``neg`` indexes the complete table; map to the evaluation slice.
    neg_local = np.searchsorted(eval_idx, neg)
    if not np.array_equal(eval_idx[neg_local], neg):
        raise RuntimeError("Negative pair left the evaluation partition")
    shards = ShardedEpochs(bags.epochs)
    post, pre = {}, {}
    for subject in SUBJECTS:
        obs = pivot[subject].to_numpy(dtype=int)
        post[subject], pre[subject] = features(shards, bags.slot_index, obs)
        post[subject] = center_scale(post[subject], train_mask)
        pre[subject] = center_scale(pre[subject], train_mask)
    report = {"design": {"fold": 0, "partition": "train only", "n_items": len(table),
                         "n_fit": int(train_mask.sum()), "n_probe": len(eval_idx),
                         "subjects": SUBJECTS, "slots": SLOTS,
                         "negative": "same book/run, n_char=n_slots=10, 5-60 rows away where possible"},
              "pairs": {}}
    for ia, sa in enumerate(SUBJECTS):
        for sb in SUBJECTS[ia + 1:]:
            key = f"{sa}_to_{sb}"
            # Fit a cross-reader linear map on other training-block items only.
            # Wide shrinkage is fixed before looking at probe scores.
            model = Ridge(alpha=1000.0, fit_intercept=False)
            model.fit(post[sa][train_mask], post[sb][train_mask])
            mapped = model.predict(post[sa][eval_idx])
            shuffled = Ridge(alpha=1000.0, fit_intercept=False)
            shuffled.fit(post[sa][train_mask], post[sb][rng.permutation(np.flatnonzero(train_mask))])
            report["pairs"][key] = {
                "post_raw_auc": paired_auc(post[sa][eval_idx], post[sb][eval_idx], neg_local),
                "post_raw_matched_win": matched_win(post[sa][eval_idx], post[sb][eval_idx], neg_local),
                "pre_raw_auc": paired_auc(pre[sa][eval_idx], pre[sb][eval_idx], neg_local),
                "post_mapped_auc": paired_auc(mapped, post[sb][eval_idx], neg_local),
                "post_mapped_matched_win": matched_win(mapped, post[sb][eval_idx], neg_local),
                "shuffled_fit_auc": paired_auc(shuffled.predict(post[sa][eval_idx]),
                                               post[sb][eval_idx], neg_local),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
