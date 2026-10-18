#!/usr/bin/env python3
"""Test held-out EEG variance attributable to row surprisal beyond nuisance.

The response is the average of *all* available physical character epochs in
four predeclared windows. The forward model sees the surprisal label only when
testing an encoding hypothesis; this is not a decoder and supplies no target
or text to any inference-time architecture. Permutations remain at item level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cogalign.io.retrospective_splits import attach_retrospective_split  # noqa: E402
from foundation.data import (CACHE_DIR, SURPRISAL_CSV, ShardedEpochs,
                             load_char_bags, load_eye_char)  # noqa: E402

OUT = ROOT / "results/response_mechanism_20261002"
WINDOWS = ((13, 39), (39, 64), (64, 115), (115, 167))
SEED = 2026


def make_features(bags, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shards = ShardedEpochs(bags.epochs)
    n = len(bags.obs)
    response = np.empty((n, 128 * len(WINDOWS)), dtype=np.float32)
    ocular = np.empty((n, 4), dtype=np.float32)
    for start in range(0, n, 64):
        stop = min(start + 64, n)
        index = [bags.slot_index[i] for i in range(start, stop)]
        counts = np.fromiter((len(ix) for ix in index), dtype=int)
        pick = np.concatenate(index)
        x = shards.take(pick)
        x -= x[..., :13].mean(axis=-1, keepdims=True)
        wave = np.stack([x[..., lo:hi].mean(axis=-1) for lo, hi in WINDOWS], axis=-1)
        # One observation is one reading. Every available slot is included;
        # later item averaging is performed only after this readout.
        sums = np.add.reduceat(wave, np.r_[0, np.cumsum(counts)[:-1]], axis=0)
        response[start:stop] = (sums / counts[:, None, None]).reshape(len(counts), -1)
        e = np.asarray(eye[pick], dtype=np.float32)
        valid = e[:, 3, :] > .5
        e_mean = np.stack([(e[:, c, :] * valid).sum(axis=1) /
                           np.maximum(valid.sum(axis=1), 1) for c in range(3)] +
                          [valid.mean(axis=1)], axis=1)
        e_sums = np.add.reduceat(e_mean, np.r_[0, np.cumsum(counts)[:-1]], axis=0)
        ocular[start:stop] = e_sums / counts[:, None]
        if start % 6400 == 0:
            print(f"[features] {stop}/{n}", flush=True)
    if not np.isfinite(response).all() or not np.isfinite(ocular).all():
        raise RuntimeError("Nonfinite response or eye covariate")
    return response, ocular


def nuisance_design(frame: pd.DataFrame, ocular: np.ndarray) -> np.ndarray:
    part = frame[["subject", "book", "run", "n_slots", "n_char", "row_idx"]].copy()
    part["book_run"] = part.book.astype(str) + "|" + part.run.astype(str)
    cat = pd.get_dummies(part[["subject", "book_run", "n_slots"]].astype(str),
                         dtype=np.float32)
    numeric = np.column_stack([part.n_char.to_numpy(dtype=float),
                               part.row_idx.to_numpy(dtype=float), ocular])
    return np.column_stack([np.ones(len(part)), cat.to_numpy(), numeric]).astype(np.float64)


def fit_nuisance(z: np.ndarray, x: np.ndarray, train: np.ndarray):
    mu = z[train, 1:].mean(axis=0)
    sd = z[train, 1:].std(axis=0).clip(min=1e-5)
    z = z.copy()
    z[:, 1:] = (z[:, 1:] - mu) / sd
    gram = z[train].T @ z[train]
    ridge = np.eye(gram.shape[0]) * 1.0
    ridge[0, 0] = 0.0
    solve = np.linalg.solve(gram + ridge, z[train].T)
    base_x = z @ (solve @ x[train].astype(np.float64))
    residual_x = x.astype(np.float64) - base_x
    return z, solve, residual_x


def stratified_permutation(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    # Each item's label is shuffled as a unit across its repeated readers.
    items = (frame.groupby("item_id", sort=False)
             .agg(book=("book", "first"), n_slots=("n_slots", "median"), y=("y", "first")))
    items["stratum"] = items.book + "|" + items.n_slots.round().astype(int).astype(str)
    perm = items.y.to_numpy().copy()
    for positions in items.groupby("stratum").indices.values():
        perm[positions] = perm[rng.permutation(positions)]
    return frame.item_id.map(dict(zip(items.index, perm))).to_numpy(dtype=np.float64)


def incremental_response(y, z, solve, residual_x, train, val):
    # Frisch-Waugh-Lovell residualization of y against the same nuisance model.
    yr = y - z @ (solve @ y[train])
    beta = (yr[train] @ residual_x[train]) / (yr[train] @ yr[train] + 1.0)
    old_sse = np.square(residual_x[val]).sum(axis=0)
    new_sse = np.square(residual_x[val] - yr[val, None] * beta).sum(axis=0)
    delta = (old_sse - new_sse) / np.maximum(old_sse, 1e-12)
    return delta.reshape(128, len(WINDOWS)), beta.reshape(128, len(WINDOWS))


def run_fold(frame, x, ocular, fold: int, permutations: int) -> dict:
    train = frame.split.eq("train").to_numpy()
    val = frame.split.eq("val").to_numpy()
    z, solve, residual_x = fit_nuisance(nuisance_design(frame, ocular), x, train)
    y = frame.y.to_numpy(dtype=np.float64)
    effect, beta = incremental_response(y, z, solve, residual_x, train, val)
    rng = np.random.default_rng(SEED + fold)
    null_max = np.empty(permutations, dtype=float)
    for p in range(permutations):
        shuffled = stratified_permutation(frame, rng)
        null_effect, _ = incremental_response(shuffled, z, solve, residual_x, train, val)
        null_max[p] = max(0.0, float(null_effect.max()))
    idx = np.unravel_index(np.argmax(effect), effect.shape)
    observed = float(effect[idx])
    p_fwer = (1 + int(np.sum(null_max >= observed))) / (permutations + 1)
    # The selected channel/window was chosen across all validation readers;
    # these reader slices are descriptive, not independent p-values.
    reader = {}
    yr = y - z @ (solve @ y[train])
    for subject in sorted(frame.subject.unique()):
        take = val & frame.subject.eq(subject).to_numpy()
        if not take.any():
            continue
        residual = residual_x[take].reshape(-1, 128, len(WINDOWS))[:, idx[0], idx[1]]
        prediction = yr[take] * beta[idx]
        reader[subject] = float((np.square(residual).sum() -
                                 np.square(residual - prediction).sum()) /
                                max(np.square(residual).sum(), 1e-12))
    np.savez_compressed(OUT / f"encoding_fold{fold}.npz", effect=effect, beta=beta,
                        null_max=null_max)
    return {"fold": fold, "n_train_readings": int(train.sum()),
            "n_val_readings": int(val.sum()),
            "n_val_items": int(frame.loc[val, "item_id"].nunique()),
            "max_delta_r2": observed, "channel_index": int(idx[0]),
            "window_index": int(idx[1]), "window_samples": WINDOWS[idx[1]],
            "fwer_permutation_p": float(p_fwer),
            "null_max_95pct": float(np.quantile(null_max, .95)),
            "positive_reader_slices": int(sum(v > 0 for v in reader.values())),
            "n_reader_slices": len(reader), "reader_delta_r2": reader}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=199)
    args = ap.parse_args()
    if args.permutations < 19:
        raise ValueError("At least 19 item-level permutations are required")
    manifest_path = OUT / "sentence_manifest.csv"
    manifest = pd.read_csv(manifest_path).fillna({"excluded_reason": ""})
    bags = load_char_bags(cache_dir=CACHE_DIR)
    eye, coverage = load_eye_char(bags)
    print(f"[eye] coverage={coverage:.4f}", flush=True)
    x, ocular = make_features(bags, eye)
    del eye
    results = []
    for fold in (0, 3):
        frame = attach_retrospective_split(bags.obs, manifest, fold,
                                           label_csv=str(SURPRISAL_CSV))
        # The third block is outside this fold's development screen and must
        # not even enter label permutations or nuisance-design construction.
        frame = frame.loc[frame.split.ne("test")].reset_index(drop=True)
        index = frame.obs_index.to_numpy(dtype=int)
        result = run_fold(frame, x[index], ocular[index], fold, args.permutations)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    passed = all(r["fwer_permutation_p"] <= .05 and
                 r["positive_reader_slices"] >= 7 for r in results)
    report = {"scope": "retrospective seven-block development, folds 0 and 3 only",
              "method": "four fixed EEG latency windows; nuisance-adjusted forward regression",
              "n_permutations": args.permutations, "windows_samples": WINDOWS,
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "positive_gate": passed, "folds": results,
              "interpretation": ("Gate for architecture screening, not a causal proof or an "
                                 "unseen-test improvement")}
    (OUT / "encoding_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"positive_gate": passed, "folds": results}, indent=2))


if __name__ == "__main__":
    main()
