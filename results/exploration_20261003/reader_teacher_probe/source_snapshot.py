#!/usr/bin/env python3
"""Training-text-only probe of a teacher excluding the student reader.

Predict count/book-residual surprisal on held training blocks using other
readers' EEG. The student baseline uses only the student reader's EEG. This
is a gate for training-only distillation, not a deployable multi-reader model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foundation.data import CACHE_DIR, SURPRISAL_CSV, ShardedEpochs, load_char_bags  # noqa: E402
from cogalign.io.retrospective_splits import attach_retrospective_split  # noqa: E402

OUT = ROOT / "results/exploration_20261003/reader_teacher_probe"
READERS = ("sub-04", "sub-05", "sub-06")
WINDOWS = ((13, 39), (39, 64), (64, 115), (115, 167))


def features(bags, indices):
    shards = ShardedEpochs(bags.epochs)
    output = []
    for lo in range(0, len(indices), 64):
        rows = indices[lo:lo + 64]
        pick = np.concatenate([
            bags.slot_index[int(i)][np.rint(np.linspace(0, len(bags.slot_index[int(i)]) - 1, 3)).astype(int)]
            for i in rows
        ])
        x = shards.take(pick).reshape(len(rows), 3, 128, 167)
        x -= x[..., :13].mean(axis=-1, keepdims=True)
        output.append(np.stack([x[..., a:b].mean(axis=(1, 3))
                                for a, b in WINDOWS], axis=-1).reshape(len(rows), -1))
    return np.concatenate(output).astype(np.float32)


def subset_train(frame, manifest, fold):
    items = frame.drop_duplicates("item_id").merge(
        manifest[["item_id", "sentence"]], on="item_id", validate="one_to_one")
    rng = np.random.default_rng(2026 + fold)
    selected = []
    for book, cap in (("GarnettDream", 1200), ("LittlePrince", 600)):
        group = items[items.book.eq(book)]
        sentences = rng.permutation(group.sentence.unique())
        total = 0
        for sentence in sentences:
            ids = group.loc[group.sentence.eq(sentence), "item_id"].tolist()
            selected.extend(ids)
            total += len(ids)
            if total >= cap:
                break
    return frame[frame.item_id.isin(selected)].copy()


def residual_labels(fit, held):
    # Labels are identical across readers, so each training item receives one
    # vote in the nuisance oracle. Held-block targets never fit this oracle.
    item = fit.groupby("item_id").agg(y=("surprisal", "first"),
                                      book=("book", "first"), n_slots=("slot_stratum", "first"))
    oracle = item.groupby(["book", "n_slots"]).y.mean()
    book_mu = item.groupby("book").y.mean()
    def get(frame):
        offset = np.array([oracle.get((r.book, r.slot_stratum), book_mu[r.book])
                           for r in frame.itertuples()], dtype=float)
        return frame.surprisal.to_numpy(dtype=float) - offset
    return get(fit), get(held)


def predict(fit, held, y, feature):
    xt = feature[fit.feature_index.to_numpy(dtype=int)]
    xv = feature[held.feature_index.to_numpy(dtype=int)]
    scaler = StandardScaler().fit(xt)
    model = Ridge(alpha=1000.).fit(scaler.transform(xt), y)
    return model.predict(scaler.transform(xv))


def score(frame):
    item = frame.groupby("item_id").agg(y=("q", "first"), pred=("pred", "mean"),
                                        book=("book", "first"), n_slots=("slot_stratum", "first"))
    strata = [(len(g), float(spearmanr(g.y, g.pred).statistic))
              for _, g in item.groupby(["book", "n_slots"])
              if len(g) >= 5 and g.y.nunique() > 1 and g.pred.nunique() > 1]
    valid = [(n, r) for n, r in strata if np.isfinite(r)]
    return {"items": len(item), "residual_item_rho": float(spearmanr(item.y, item.pred).statistic),
            "conditional_rho": float(np.average([r for _, r in valid],
                                                 weights=[n for n, _ in valid]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bags = load_char_bags(cache_dir=CACHE_DIR)
    manifest = pd.read_csv(ROOT / "results/response_mechanism_20261002/sentence_manifest.csv")
    frames = {}
    for fold in (0, 3):
        frame = attach_retrospective_split(bags.obs, manifest, fold,
                                           label_csv=str(SURPRISAL_CSV))
        frame = frame[frame.split.eq("train")].copy()
        frame["slot_stratum"] = frame.groupby("item_id").n_slots.transform("median")
        frames[fold] = subset_train(frame, manifest, fold)
    indices = np.unique(np.concatenate([f.obs_index.to_numpy(dtype=int) for f in frames.values()]))
    feature = features(bags, indices)
    lookup = dict(zip(indices, np.arange(len(indices))))
    rows, predictions = [], []
    for fold, frame in frames.items():
        frame["feature_index"] = frame.obs_index.map(lookup)
        for held_block in sorted(frame.block.unique()):
            fit = frame[frame.block.ne(held_block)]
            held = frame[frame.block.eq(held_block)]
            qfit, qheld = residual_labels(fit, held)
            fit = fit.assign(q=qfit); held = held.assign(q=qheld)
            for reader in READERS:
                teacher_fit = fit[fit.subject.ne(reader)]
                teacher_held = held[held.subject.ne(reader)]
                student_fit = fit[fit.subject.eq(reader)]
                student_held = held[held.subject.eq(reader)]
                # Every compared item needs both the student and other readers.
                common = set(student_held.item_id) & set(teacher_held.item_id)
                teacher_held = teacher_held[teacher_held.item_id.isin(common)].copy()
                student_held = student_held[student_held.item_id.isin(common)].copy()
                teacher_held["pred"] = predict(teacher_fit, teacher_held,
                                               teacher_fit.q.to_numpy(), feature)
                student_held["pred"] = predict(student_fit, student_held,
                                               student_fit.q.to_numpy(), feature)
                # A mismatched teacher retains book/count and held-block structure.
                teacher_item = teacher_held.groupby("item_id").agg(
                    pred=("pred", "mean"), book=("book", "first"),
                    slot_stratum=("slot_stratum", "first"))
                rng = np.random.default_rng(2026 + 100 * fold + int(held_block))
                shuffled = teacher_item.pred.copy()
                for _, group in teacher_item.groupby(["book", "slot_stratum"]):
                    shuffled.loc[group.index] = rng.permutation(group.pred.to_numpy())
                mismatch = teacher_held.copy()
                mismatch["pred"] = mismatch.item_id.map(shuffled)
                for name, result in (("other_reader_teacher", teacher_held),
                                     ("single_reader_baseline", student_held),
                                     ("mismatched_teacher", mismatch)):
                    value = score(result)
                    rows.append({"outer_development_fold": fold, "held_training_block": int(held_block),
                                 "student_reader": reader, "arm": name, **value})
                    predictions.append(result[["subject", "book", "run", "row_idx", "item_id", "slot_stratum", "q", "pred"]]
                                       .assign(outer_development_fold=fold,
                                               held_training_block=int(held_block), student_reader=reader, arm=name))
            print(f"[probe] development fold={fold} held training block={held_block}", flush=True)
    scores = pd.DataFrame(rows)
    means = scores.groupby(["outer_development_fold", "arm"]).conditional_rho.mean().unstack()
    gate = bool(((means.other_reader_teacher > .05) &
                 (means.other_reader_teacher > means.single_reader_baseline) &
                 (means.other_reader_teacher > means.mismatched_teacher)).all())
    scores.to_csv(OUT / "inner_block_scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "training_block_predictions.csv", index=False)
    report = {"scope": "only training texts in each fixed development fold; teacher excludes student reader",
              "ridge_alpha": 1000., "physical_slots_per_reading": 3,
              "student_readers": READERS, "teacher_gate_passed": gate,
              "mean_conditional_rho": means.reset_index().to_dict(orient="records"),
              "limitation": "Fixed raw-window Ridge probe; not a test of every possible trained neural teacher"}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
