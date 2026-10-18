"""Character-bag dataset over the character-locked EEG cache.

One training example is one *reading* of one text row by one participant: a
variable-length bag of fixed-length character epochs on the physical time axis.
The label is the row's language-model surprisal, identical to the frozen JBHI
experiment, so the numbers stay directly comparable.
"""
from __future__ import annotations

import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REF_SRC = Path(__file__).resolve().parents[1]
if str(REF_SRC) not in sys.path:
    sys.path.insert(0, str(REF_SRC))

from cogalign.io.splits import assign_blocked_splits, check_no_item_leakage  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("SURPRISAL_EEG_CACHE", PACKAGE_ROOT / "data_cache" / "char_epochs_1200ms"))
# The exact label file of the frozen run: uer/gpt2-chinese-cluecorpussmall with
# a 192-token context window (md5 of "model|ctx192" is the cache tag).
SURPRISAL_CSV = Path(os.environ.get("SURPRISAL_LABEL_CSV", PACKAGE_ROOT / "data_cache" / "surprisal_1829a99ad4.csv"))
SUBJECTS = ("sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
            "sub-09", "sub-10", "sub-13", "sub-14", "sub-15")
BOOKS = ("LittlePrince", "GarnettDream")
BASELINE_SAMPLES = 13  # -100..0 ms at 128 Hz


@dataclass
class CharBags:
    """All character epochs plus the row-observation index over them."""

    epochs: np.ndarray            # [n_chars, 128, 71] float16, microvolts
    group_id: np.ndarray          # [n_chars] int, index into ``groups``
    groups: tuple[str, ...]       # "subject|book" recording keys
    obs: pd.DataFrame             # one row per (subject, row) observation
    slot_index: list[np.ndarray]  # obs -> indices into ``epochs``


def load_char_bags(
    *,
    cache_dir: Path = CACHE_DIR,
    subjects: tuple[str, ...] = SUBJECTS,
    books: tuple[str, ...] = BOOKS,
    mmap: bool = True,
) -> CharBags:
    arrays: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []
    groups: list[str] = []
    group_ids: list[np.ndarray] = []
    offset = 0
    for subject in subjects:
        for book in books:
            x_path = cache_dir / f"X_{subject}_{book}.npy"
            m_path = cache_dir / f"meta_{subject}_{book}.csv"
            if not x_path.exists() or not m_path.exists():
                continue
            array = np.load(x_path, mmap_mode="r" if mmap else None)
            meta = pd.read_csv(m_path)
            if len(meta) != len(array):
                raise RuntimeError(f"Cache length mismatch for {subject} {book}")
            meta["global_index"] = np.arange(offset, offset + len(meta))
            arrays.append(array)
            metas.append(meta)
            group_ids.append(np.full(len(meta), len(groups), dtype=np.int64))
            groups.append(f"{subject}|{book}")
            offset += len(meta)
    if not arrays:
        raise FileNotFoundError(f"No character cache found under {cache_dir}")

    meta = pd.concat(metas, ignore_index=True)
    group_id = np.concatenate(group_ids)

    obs_keys = ["subject", "book", "run", "row_idx"]
    order = meta.sort_values(obs_keys + ["char_slot"])
    slot_index: list[np.ndarray] = []
    obs_records: list[dict] = []
    for key, chunk in order.groupby(obs_keys, sort=True):
        slot_index.append(chunk["global_index"].to_numpy())
        obs_records.append(
            {
                "subject": key[0],
                "book": key[1],
                "run": int(key[2]),
                "row_idx": int(key[3]),
                "text": str(chunk["text"].iloc[0]),
                "n_slots": int(len(chunk)),
                "group_id": int(group_id[chunk["global_index"].iloc[0]]),
                # audit-only
                "n_char": int(chunk["n_char"].iloc[0]),
                "dur": float(chunk["dur"].iloc[0]),
            }
        )
    obs = pd.DataFrame(obs_records)
    return CharBags(
        epochs=arrays if mmap else np.concatenate(arrays),
        group_id=group_id,
        groups=tuple(groups),
        obs=obs,
        slot_index=slot_index,
    )


class ShardedEpochs:
    """Read-only view that indexes a list of memory-mapped shards as one array."""

    def __init__(self, shards: list[np.ndarray]) -> None:
        self.shards = shards
        self.bounds = np.cumsum([0] + [len(s) for s in shards])

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def take(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        which = np.searchsorted(self.bounds, indices, side="right") - 1
        out = np.empty((len(indices),) + self.shards[0].shape[1:], dtype=np.float32)
        for shard_id in np.unique(which):
            mask = which == shard_id
            local = indices[mask] - self.bounds[shard_id]
            out[mask] = np.asarray(self.shards[shard_id][local], dtype=np.float32)
        return out


def attach_labels_and_splits(obs: pd.DataFrame, fold: int, n_blocks: int = 7,
                             label_csv: Path | str | None = None) -> pd.DataFrame:
    """Attach the surprisal label and reproduce the frozen block split.

    ``label_csv`` defaults to the frozen run's own label file. Any replacement
    must use the same ``book, run, row_idx, surprisal`` schema so the split,
    the standardisation and the tercile edges are computed identically.
    """
    labels = pd.read_csv(label_csv or SURPRISAL_CSV)
    labels = labels[["book", "run", "row_idx", "surprisal"]].copy()
    labels["run"] = labels["run"].astype(int)
    labels["row_idx"] = labels["row_idx"].astype(int)

    merged = obs.merge(labels, on=["book", "run", "row_idx"], how="left")
    merged = merged[np.isfinite(merged["surprisal"].to_numpy(dtype=float))].reset_index(drop=True)

    merged = assign_blocked_splits(merged, fold=fold, n_blocks=n_blocks)
    check_no_item_leakage(merged)

    raw = merged["surprisal"].to_numpy(dtype=float)
    train = merged["split"].to_numpy() == "train"
    mean, std = float(raw[train].mean()), float(raw[train].std()) + 1e-8
    merged["y"] = (raw - mean) / std
    merged.attrs["target_mean"] = mean
    merged.attrs["target_std"] = std

    # Train-only terciles drive the high-vs-low item AUC endpoint.
    lo, hi = np.quantile(raw[train], [1 / 3, 2 / 3])
    merged["tercile"] = np.digitize(raw, [lo, hi])
    return merged


class CharBagDataset(Dataset):
    """Yields one padded bag of character epochs per row observation."""

    def __init__(
        self,
        shards: ShardedEpochs,
        slot_index: list[np.ndarray],
        frame: pd.DataFrame,
        *,
        max_slots: int = 10,
        char_targets: np.ndarray | None = None,
    ) -> None:
        self.shards = shards
        self.slot_index = slot_index
        self.frame = frame.reset_index(drop=True)
        self.max_slots = int(max_slots)
        self.positions = self.frame["obs_index"].to_numpy()
        self.y = self.frame["y"].to_numpy(dtype=np.float32)
        self.group = self.frame["group_id"].to_numpy(dtype=np.int64)
        self.char_targets = (
            np.zeros((len(self.frame), self.max_slots), dtype=np.float32)
            if char_targets is None
            else np.asarray(char_targets, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        indices = self.slot_index[int(self.positions[i])][: self.max_slots]
        epochs = self.shards.take(indices)
        n = len(epochs)
        padded = np.zeros((self.max_slots,) + epochs.shape[1:], dtype=np.float32)
        padded[:n] = epochs
        mask = np.zeros(self.max_slots, dtype=np.float32)
        mask[:n] = 1.0
        return {
            "eeg": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor(self.y[i], dtype=torch.float32),
            "y_char": torch.from_numpy(self.char_targets[i]),
            "group": torch.tensor(self.group[i], dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }


CHAR_SURPRISAL_CSV = Path(os.environ.get("SURPRISAL_CHAR_LABEL_CSV", PACKAGE_ROOT / "data_cache" / "char_surprisal.csv"))


def load_char_targets(frame: pd.DataFrame, slot_index: list, meta: pd.DataFrame,
                      max_slots: int = 10, *, physical_slots: np.ndarray | None = None,
                      cache_dir: Path = CACHE_DIR, return_validity: bool = False):
    """Align slot contributions whose mean reproduces the frozen ROW target.

    Stored character values include folded punctuation NLL. Their SUM divided
    by token count is the row target; their slot MEAN is not. Multiply by
    n_slots/n_tokens, then distribute the <=0.01-nat fp16 discrepancy uniformly
    to reproduce the frozen scalar exactly. Missing physical slots invalidate
    local/conservation supervision, while ordinary row supervision is kept.
    """
    table = pd.read_csv(CHAR_SURPRISAL_CSV)
    table["run"] = table["run"].astype(int)
    table["row_idx"] = table["row_idx"].astype(int)
    mean = float(frame.attrs["target_mean"])
    std = float(frame.attrs["target_std"])
    if physical_slots is None:
        # Same recording order as load_char_bags, and thus slot_index's global
        # epoch indices. Never infer slot identity from the bag length.
        group_keys = list(meta[["subject", "book", "group_id"]].drop_duplicates()
                          .sort_values("group_id").itertuples(index=False, name=None))
        physical_slots = np.concatenate([
            pd.read_csv(cache_dir / f"meta_{subject}_{book}.csv",
                        usecols=["char_slot"]).char_slot.to_numpy(dtype=int)
            for subject, book, _ in group_keys
        ])
    row_lookup = {(r.book, int(r.run), int(r.row_idx)): r
                  for r in frame.drop_duplicates(["book", "run", "row_idx"]).itertuples(index=False)}
    lookup = {}
    for key, chunk in table.groupby(["book", "run", "row_idx"], sort=False):
        if key not in row_lookup:
            continue
        group = chunk.sort_values("char_slot")
        row = row_lookup[key]
        if group.n_tokens.nunique() != 1 or group.char_slot.duplicated().any():
            raise RuntimeError("Inconsistent character label schema")
        raw = group.char_surprisal.to_numpy(dtype=float)
        token_count = int(group.n_tokens.iloc[0])
        correction = float(row.surprisal) - float(raw.sum() / token_count)
        if abs(correction) > .01:
            raise RuntimeError("Local labels differ from the frozen row target by >0.01 nats")
        expected = [c for c in str(row.text)
                    if not unicodedata.category(c).startswith("P") and not c.isspace()]
        # Old cache generation truncates unknown/multichar tokenizer strings
        # and creates a leading-punctuation pseudo-slot. Those rows cannot be
        # given physical local labels from this table; retain row loss only.
        if group["char"].astype(str).tolist() != expected:
            lookup[key] = None
            continue
        values = (raw * len(raw) / token_count + correction - mean) / std
        lookup[key] = dict(zip(group.char_slot.to_numpy(dtype=int), values))
    out = np.zeros((len(frame), max_slots), dtype=np.float32)
    valid = np.zeros(len(frame), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        key = (row.book, int(row.run), int(row.row_idx))
        if key not in lookup:
            raise RuntimeError(f"Missing character labels for {row.book}/{row.run}/{row.row_idx}")
        by_slot = lookup[key]
        if by_slot is None:
            continue
        indices = np.asarray(slot_index[int(row.obs_index)])[:max_slots]
        actual = np.asarray(physical_slots)[indices]
        if any(int(slot) not in by_slot for slot in actual):
            raise RuntimeError("An EEG slot has no matching local label")
        out[i, :len(actual)] = [by_slot[int(slot)] for slot in actual]
        complete = set(actual) == set(by_slot) and len(actual) == len(by_slot)
        valid[i] = float(complete)
        if complete and abs(float(out[i, :len(actual)].mean()) - float(row.y)) > 1e-5:
            raise RuntimeError("Corrected character contributions violate row identity")
    if not return_validity and not valid.all():
        raise RuntimeError("Incomplete bags require return_validity=True to mask local supervision")
    return (out, valid) if return_validity else out


def baseline_correct(x: torch.Tensor, n: int = BASELINE_SAMPLES) -> torch.Tensor:
    """Subtract the pre-onset mean of each channel; x is [..., channels, time]."""
    return x - x[..., :n].mean(dim=-1, keepdim=True)


ROW_FROZEN_DIR = Path(os.environ.get("SURPRISAL_ROW_CACHE", PACKAGE_ROOT / "data_cache" / "epochs"))
ROW_AMP_DIR = Path(os.environ.get("SURPRISAL_ROW_AMP_CACHE", PACKAGE_ROOT / "data_cache" / "row_epochs_amp"))


def load_row_bags(
    *,
    source: str,
    subjects: tuple[str, ...] = SUBJECTS,
    books: tuple[str, ...] = BOOKS,
) -> CharBags:
    """Row-level epochs presented as one-slot bags.

    ``source="frozen"`` reads the sealed row cache the frozen experiment was
    trained on -- warped to 128 samples *and* per-epoch MAD normalised.
    ``source="amp"`` reads the rebuilt cache that warps identically but keeps
    microvolts. Running both through the same encoder as the character arms is
    what separates the time axis from the amplitude handling.
    """
    if source not in {"frozen", "amp"}:
        raise ValueError("source must be 'frozen' or 'amp'")
    arrays, metas, groups, group_ids = [], [], [], []
    offset = 0
    for subject in subjects:
        for book in books:
            if source == "frozen":
                path = ROW_FROZEN_DIR / f"epochs_{subject}_{book}_0.5_30.npz"
                if not path.exists():
                    continue
                blob = np.load(path, allow_pickle=True)
                array = blob["X"]
                meta = pd.DataFrame(blob["meta"])
            else:
                x_path = ROW_AMP_DIR / f"X_{subject}_{book}.npy"
                m_path = ROW_AMP_DIR / f"meta_{subject}_{book}.csv"
                if not x_path.exists() or not m_path.exists():
                    continue
                array = np.load(x_path, mmap_mode="r")
                meta = pd.read_csv(m_path)
            meta = meta.rename(columns={"session": "book"})
            meta["subject"] = subject
            meta["book"] = book
            meta["run"] = meta["run"].astype(int)
            meta["row_idx"] = meta["row_idx"].astype(int)
            meta["text"] = meta["text"].astype(str)
            if len(meta) != len(array):
                raise RuntimeError(f"Row cache length mismatch for {subject} {book}")
            meta["global_index"] = np.arange(offset, offset + len(meta))
            arrays.append(array)
            metas.append(meta)
            group_ids.append(np.full(len(meta), len(groups), dtype=np.int64))
            groups.append(f"{subject}|{book}")
            offset += len(meta)
    if not arrays:
        raise FileNotFoundError(f"No row cache found for source={source!r}")

    meta = pd.concat(metas, ignore_index=True)
    group_id = np.concatenate(group_ids)
    keys = ["subject", "book", "run", "row_idx"]
    order = meta.sort_values(keys)
    slot_index = [np.array([i], dtype=np.int64) for i in order["global_index"].to_numpy()]
    obs = pd.DataFrame({
        "subject": order["subject"].to_numpy(),
        "book": order["book"].to_numpy(),
        "run": order["run"].to_numpy(),
        "row_idx": order["row_idx"].to_numpy(),
        "text": order["text"].to_numpy(),
        "n_slots": 1,
        "group_id": group_id[order["global_index"].to_numpy()],
        "n_char": order["n_char"].to_numpy(),
        "dur": order["dur"].to_numpy(),
    }).reset_index(drop=True)
    return CharBags(epochs=arrays, group_id=group_id, groups=tuple(groups),
                    obs=obs, slot_index=slot_index)


def per_epoch_mad(x: torch.Tensor) -> torch.Tensor:
    """Median/MAD scaling of each epoch, as the frozen builder applied it.

    This is the operation that discards cross-row ERP amplitude, so it is a
    factor in the attribution design rather than a preprocessing default.
    """
    median = x.median(dim=-1, keepdim=True).values
    mad = (x - median).abs().median(dim=-1, keepdim=True).values
    # A 71-sample character epoch can contain a near-flat channel whose MAD is
    # essentially zero; a fixed 1e-6 floor then divides by nothing and the whole
    # epoch explodes to inf. Floor the scale by the epoch's own spread instead.
    floor = 1e-3 * x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - median) / (1.4826 * mad).clamp_min(floor)


ROW_SEQ_DIR = Path(os.environ.get("SURPRISAL_ROW_SEQUENCE_CACHE", PACKAGE_ROOT / "data_cache" / "row_sequence"))


def load_row_sequences(*, cache_dir: Path = ROW_SEQ_DIR,
                       subjects: tuple[str, ...] = SUBJECTS,
                       books: tuple[str, ...] = BOOKS) -> CharBags:
    """One padded continuous sequence per row, presented as a one-slot bag.

    The slot axis carries a single sequence rather than a character, so the same
    batching and spatial-transform machinery applies unchanged; the per-sample
    valid length travels in ``obs.length`` and becomes the time mask.
    """
    arrays, metas, groups, group_ids = [], [], [], []
    offset = 0
    for subject in subjects:
        for book in books:
            x_path = cache_dir / f"X_{subject}_{book}.npy"
            m_path = cache_dir / f"meta_{subject}_{book}.csv"
            if not x_path.exists() or not m_path.exists():
                continue
            array = np.load(x_path, mmap_mode="r")
            meta = pd.read_csv(m_path)
            if len(meta) != len(array):
                raise RuntimeError(f"Row-sequence length mismatch for {subject} {book}")
            meta["global_index"] = np.arange(offset, offset + len(meta))
            arrays.append(array); metas.append(meta)
            group_ids.append(np.full(len(meta), len(groups), dtype=np.int64))
            groups.append(f"{subject}|{book}")
            offset += len(meta)
    if not arrays:
        raise FileNotFoundError(f"No row-sequence cache under {cache_dir}")
    meta = pd.concat(metas, ignore_index=True)
    group_id = np.concatenate(group_ids)
    keys = ["subject", "book", "run", "row_idx"]
    order = meta.sort_values(keys)
    slot_index = [np.array([i], dtype=np.int64) for i in order["global_index"].to_numpy()]
    obs = order[keys + ["text", "n_char", "dur", "length"]].copy()
    obs["n_slots"] = 1
    obs["group_id"] = group_id[order["global_index"].to_numpy()]
    return CharBags(epochs=arrays, group_id=group_id, groups=tuple(groups),
                    obs=obs.reset_index(drop=True), slot_index=slot_index)


EYE_CHAR_DIR = Path(os.environ.get("SURPRISAL_EYE_CACHE", PACKAGE_ROOT / "data_cache" / "eye_char"))


def load_eye_char(bags: CharBags, *, cache_dir: Path = EYE_CHAR_DIR,
                  char_cache: Path | None = None) -> tuple[np.ndarray, float]:
    """Eye epochs aligned one-to-one with the character epochs of ``bags``.

    The join is on (subject, book, run, row_idx, char_slot) and is exact where
    eye data exists: a silent mismatch would pair one character's EEG with
    another character's gaze, which no downstream metric would reveal.

    About 4.4% of character epochs have no eye counterpart, and the cause is the
    dataset rather than this pipeline: sub-08 LittlePrince runs 1-2 ship no
    eye-tracking archive at all, and seven sub-09 GarnettDream archives are
    truncated (348 MB on disk against a 1.1 GB header, with no readable
    gazedata member). Those epochs are zero-filled with the validity channel at
    zero rather than dropped. Dropping them would change the evaluation set and
    break comparability with every other arm; zero-filling keeps the rows and
    lets the validity channel say the modality is absent, which is what the
    zero-initialised fusion gate is there to exploit.

    Returns the tensor and the fraction of epochs that carry real eye data.
    """
    frames = []
    for group in bags.groups:
        subject, book = group.split("|")
        path = cache_dir / f"meta_{subject}_{book}.csv"
        array = cache_dir / f"X_{subject}_{book}.npy"
        if not path.exists() or not array.exists():
            raise FileNotFoundError(f"Eye cache missing for {group}")
        meta = pd.read_csv(path)
        meta["_file"] = str(array)
        meta["_row"] = np.arange(len(meta))
        frames.append(meta)
    eye_meta = pd.concat(frames, ignore_index=True)

    keys = ["subject", "book", "run", "row_idx", "char_slot"]
    # Read the char metadata from the cache the bags actually came from; the
    # 450 ms and 900 ms caches differ by a handful of boundary epochs.
    source_dir = Path(char_cache) if char_cache else CACHE_DIR
    char_meta = pd.concat(
        [pd.read_csv(source_dir / f"meta_{g.split('|')[0]}_{g.split('|')[1]}.csv",
                     usecols=keys)
         for g in bags.groups],
        ignore_index=True,
    )
    char_meta["_order"] = np.arange(len(char_meta))
    merged = char_meta.merge(eye_meta[keys + ["_file", "_row"]], on=keys, how="left")
    present = merged["_row"].notna()

    first = np.load(eye_meta["_file"].iloc[0], mmap_mode="r")
    out = np.zeros((len(merged),) + first.shape[1:], dtype=np.float16)
    for path, chunk in merged[present].groupby("_file", sort=False):
        source = np.load(path, mmap_mode="r")
        out[chunk["_order"].to_numpy()] = source[chunk["_row"].to_numpy().astype(int)]
    return out, float(present.mean())
