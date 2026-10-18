#!/usr/bin/env python3
"""Train/validation-only probe for induced EEG band-power surprisal signal.

Uses four physical character epochs spread over each row, never stimulus
identity, eye data, item index, duration, or target-side language features.
The probe is deliberately small; it gates whether to train a new spectral
branch, and is not a replacement for the full decoder.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foundation.data import ShardedEpochs, attach_labels_and_splits, load_char_bags  # noqa: E402

CACHE = Path(os.environ.get("SURPRISAL_EEG_CACHE", "/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms"))
SOURCE = ROOT / "results/overlap_demix_validation_20260930/tables/charbag_baseline_val_predictions.csv"
OUT = ROOT / "results/spectral_probe_20261002"
SUBJECTS = ("sub-04", "sub-05", "sub-06")
SEED = 20261002
BANDS = ((4, 8), (8, 13), (13, 30))


def spectral_features(shards: ShardedEpochs, slot_index: list[np.ndarray],
                      obs_indices: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(obs_indices), 32):
        rows = obs_indices[start:start + 32]
        indices = []
        for row in rows:
            slots = slot_index[int(row)]
            positions = np.rint(np.linspace(0, len(slots) - 1, 4)).astype(int)
            indices.extend(slots[positions])
        x = shards.take(np.asarray(indices)).reshape(len(rows), 4, 128, 167)
        # Correct the pre-onset baseline; the two 0.5 s windows then probe
        # early and late induced activity, averaged over four physical slots.
        x = x - x[..., :13].mean(axis=-1, keepdims=True)
        parts = []
        for lo, hi in ((13, 77), (77, 141)):
            wave = x[..., lo:hi] * np.hanning(hi - lo).astype(np.float32)
            power = np.abs(np.fft.rfft(wave, axis=-1)) ** 2
            freq = np.fft.rfftfreq(hi - lo, 1 / 128)
            for fmin, fmax in BANDS:
                band = np.log1p(power[..., (freq >= fmin) & (freq < fmax)].mean(axis=-1))
                parts.append(band.mean(axis=1))
        output.append(np.concatenate(parts, axis=1).astype(np.float32))
    return np.concatenate(output)


def item_score(frame: pd.DataFrame) -> dict[str, float]:
    items = frame.groupby("item_id").agg(y=("y", "first"), pred=("pred", "mean"),
                                         book=("book", "first"), n_slots=("n_slots", "first"))
    strata = [(len(group), float(spearmanr(group.y, group.pred).statistic))
              for _, group in items.groupby(["book", "n_slots"]) if len(group) >= 5]
    return {"items": len(items),
            "item_rho": float(spearmanr(items.y, items.pred).statistic),
            "book_count_rho": float(np.average([rho for _, rho in strata],
                                               weights=[n for n, _ in strata]))}


def main() -> None:
    rng = np.random.default_rng(SEED)
    bags = load_char_bags(cache_dir=CACHE, subjects=SUBJECTS)
    frame = bags.obs.copy()
    frame["obs_index"] = np.arange(len(frame))
    frame = attach_labels_and_splits(frame, fold=0)
    frame = frame[(frame.n_slots >= 4) & frame.subject.isin(SUBJECTS)].copy()
    # Cap by item, independent of surprisal. All readings of a chosen item stay
    # in its assigned train or validation block.
    selected = []
    for split, book, cap in (("train", "GarnettDream", 1200),
                             ("train", "LittlePrince", 600),
                             ("val", "GarnettDream", 400),
                             ("val", "LittlePrince", 200)):
        ids = frame.loc[(frame.split == split) & (frame.book == book), "item_id"].unique()
        selected.extend(rng.choice(ids, size=min(cap, len(ids)), replace=False))
    frame = frame[frame.item_id.isin(selected)].sort_values(["split", "item_id", "subject"]).reset_index(drop=True)
    x = spectral_features(ShardedEpochs(bags.epochs), bags.slot_index,
                          frame.obs_index.to_numpy(dtype=int))
    train = (frame.split == "train").to_numpy()
    valid = (frame.split == "val").to_numpy()
    y = frame.y.to_numpy(dtype=np.float32)
    # Remove recording-specific average power using training observations
    # only. This prevents a reader/book amplitude offset from driving the fit.
    group = frame.group_id.to_numpy()
    for gid in np.unique(group):
        fit = train & (group == gid)
        take = group == gid
        mu = x[fit].mean(axis=0)
        sd = x[fit].std(axis=0).clip(min=1e-3)
        x[take] = (x[take] - mu) / sd
    scaler = StandardScaler().fit(x[train])
    x_train = scaler.transform(x[train])
    x_val = scaler.transform(x[valid])
    model = Ridge(alpha=1000.0)
    model.fit(x_train, y[train])
    probe = frame.loc[valid, ["subject", "book", "run", "row_idx", "item_id", "n_slots", "y"]].copy()
    probe["pred"] = model.predict(x_val)
    # A shuffled-label fit in the same book/count groups detects any
    # deterministic signal unrelated to the training label assignment.
    shuffled = y[train].copy()
    train_rows = frame.loc[train]
    for _, positions in train_rows.groupby(["book", "n_slots"]).indices.items():
        shuffled[positions] = shuffled[rng.permutation(positions)]
    control = Ridge(alpha=1000.0).fit(x_train, shuffled)
    null = probe.copy()
    null["pred"] = control.predict(x_val)
    baseline = pd.read_csv(SOURCE)
    baseline = baseline[baseline.fold == 0].merge(
        probe[["subject", "book", "run", "row_idx", "n_slots"]],
        on=["subject", "book", "run", "row_idx"], how="inner", validate="one_to_one")
    scores = {"design": {"fold": 0, "subjects": SUBJECTS,
                         "n_train_readings": int(train.sum()),
                         "n_val_readings": int(valid.sum()),
                         "bands_hz": BANDS, "ridge_alpha": 1000.0,
                         "slots_per_row": 4},
              "spectral": item_score(probe),
              "shuffled_target": item_score(null),
              "matched_neural_baseline": item_score(baseline)}
    OUT.mkdir(parents=True, exist_ok=True)
    probe.to_csv(OUT / "spectral_val_predictions.csv", index=False)
    (OUT / "probe.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
