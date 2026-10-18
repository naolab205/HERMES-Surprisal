"""Retrospective sentence-disjoint, context-purged ChineseEEG partitions.

Each recording is divided once at a sentence boundary. The two halves are
assigned to different book-stratified folds. Rows near that boundary are
excluded so that a 192-token LM context and a 1.2-s EEG epoch cannot reach
another partition. Repeated or near-duplicate sentences assigned to distinct
folds are excluded in their entirety, across every reader.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import re
import unicodedata

import numpy as np
import pandas as pd

from .splits import add_item_ids


END = re.compile(r"[。！？!?]")


def normalize_sentence(value: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", value).casefold()
                   if ch.isalnum())


def audit_cross_block_near_duplicates(manifest: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Broader read-only four-gram audit of retained sentence families."""
    retained = manifest[manifest.included].drop_duplicates(["book", "sentence", "block"])
    found = []
    for book, group in retained.groupby("book"):
        values = list(group[["sentence", "block"]].itertuples(index=False, name=None))
        inverted: dict[str, list[int]] = defaultdict(list)
        for i, (sentence, _) in enumerate(values):
            if len(sentence) >= 12:
                for token in {sentence[k:k + 4] for k in range(len(sentence) - 3)}:
                    inverted[token].append(i)
        for i, (a, block) in enumerate(values):
            if len(a) < 12:
                continue
            grams = {a[k:k + 4] for k in range(len(a) - 3)}
            candidates = Counter(j for token in grams for j in inverted[token] if j > i)
            for j, overlap in candidates.items():
                b, other_block = values[j]
                if block == other_block or overlap < .55 * min(len(a), len(b)):
                    continue
                if min(len(a), len(b)) / max(len(a), len(b)) < .85:
                    continue
                if SequenceMatcher(None, a, b, autojunk=False).ratio() >= .90:
                    found.append((str(book), a, b))
    return found


def sentence_manifest(obs: pd.DataFrame, *, n_blocks: int = 7,
                      purge_rows: int = 24) -> pd.DataFrame:
    if n_blocks != 7 or purge_rows < 24:
        raise ValueError("This registered design uses seven folds and a >=24-row purge")
    required = {"book", "run", "row_idx", "text"}
    if not required.issubset(obs):
        raise ValueError(f"Missing columns: {required - set(obs)}")
    unique = (add_item_ids(obs).drop_duplicates(["book", "run", "row_idx"])
              .sort_values(["book", "run", "row_idx"]).reset_index(drop=True))
    if unique.duplicated("item_id").any():
        raise ValueError("One item ID maps to multiple physical rows")
    records: list[dict] = []
    book_piece_count: dict[str, int] = defaultdict(int)
    for (book, run), group in unique.groupby(["book", "run"], sort=True):
        rows = list(group.itertuples(index=False))
        sentences: list[list] = []
        current: list = []
        for row in rows:
            current.append(row)
            if END.search(str(row.text)):
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
        if len(sentences) < 2:
            raise ValueError(f"Cannot split {book} run {run} into two sentences")
        lengths = np.cumsum([len(s) for s in sentences])
        cut = int(np.argmin(np.abs(lengths[:-1] - len(rows) / 2))) + 1
        sentence_index = 0
        for half, part in enumerate((sentences[:cut], sentences[cut:])):
            block = book_piece_count[str(book)] % n_blocks
            book_piece_count[str(book)] += 1
            for sentence in part:
                norm = normalize_sentence("".join(str(r.text) for r in sentence))
                for row in sentence:
                    # The row-index distance is deliberately conservative: 24
                    # ordinary rows cover at least a 192-token context even if
                    # every row contains roughly ten target characters.
                    records.append({"item_id": row.item_id, "book": book,
                                    "run": int(run), "row_idx": int(row.row_idx),
                                    "sentence": norm, "sentence_index": sentence_index,
                                    "block": block, "half": half,
                                    "excluded_reason": ""})
                sentence_index += 1
        # Exclude the same number of rows on both sides of the one within-run
        # split; the gap also covers neighbouring 1.2-s EEG windows.
        run_records = records[-len(rows):]
        first_right = next(i for i, rec in enumerate(run_records) if rec["half"] == 1)
        for rec in run_records[max(0, first_right - purge_rows):
                               min(len(rows), first_right + purge_rows)]:
            rec["excluded_reason"] = "context_or_eeg_boundary"
    out = pd.DataFrame(records)

    # Detect duplicates before boundary removal. Group by book: the two books
    # are separate continuous contexts even when a short phrase is identical.
    by_text = out.groupby(["book", "sentence"])["block"].nunique()
    conflicts = set(by_text[by_text > 1].index)
    sentence_keys = sorted(set(zip(out.book, out.sentence)))
    candidates: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for book, sentence in sentence_keys:
        if len(sentence) >= 12:
            candidates[(book, sentence[:6], len(sentence) // 4)].append(sentence)
    for (book, prefix, bucket), values in candidates.items():
        for other_bucket in (bucket - 1, bucket, bucket + 1):
            for a in values:
                for b in candidates.get((book, prefix, other_bucket), []):
                    if a >= b or min(len(a), len(b)) / max(len(a), len(b)) < .9:
                        continue
                    if SequenceMatcher(None, a, b, autojunk=False).ratio() >= .92:
                        a_blocks = set(out.loc[(out.book == book) & (out.sentence == a), "block"])
                        b_blocks = set(out.loc[(out.book == book) & (out.sentence == b), "block"])
                        if a_blocks != b_blocks:
                            conflicts.update(((book, a), (book, b)))
    duplicate_mask = pd.Series(list(zip(out.book, out.sentence)), index=out.index).isin(conflicts)
    out.loc[duplicate_mask, "excluded_reason"] = "cross_block_duplicate_sentence"
    out["included"] = out.excluded_reason.eq("")
    if not out.loc[out.included].groupby("book").block.nunique().eq(n_blocks).all():
        raise RuntimeError("The purge left a book absent from at least one fold")
    return out


def attach_retrospective_split(obs: pd.DataFrame, manifest: pd.DataFrame,
                               fold: int, *, label_csv: str) -> pd.DataFrame:
    if not 0 <= fold < 7:
        raise ValueError("fold must be in 0..6")
    base = add_item_ids(obs)
    base["obs_index"] = np.arange(len(base))
    included = manifest.loc[manifest.included, ["item_id", "block"]]
    frame = base.merge(included, on="item_id", how="inner", validate="many_to_one")
    labels = pd.read_csv(label_csv, usecols=["book", "run", "row_idx", "surprisal"])
    frame = frame.merge(labels, on=["book", "run", "row_idx"], how="left", validate="many_to_one")
    if frame.surprisal.isna().any():
        raise RuntimeError("Missing frozen surprisal labels")
    frame["split"] = "train"
    frame.loc[frame.block.eq((fold + 1) % 7), "split"] = "val"
    frame.loc[frame.block.eq(fold), "split"] = "test"
    raw = frame.surprisal.to_numpy(dtype=float)
    train = frame.split.eq("train").to_numpy()
    mean, std = float(raw[train].mean()), float(raw[train].std()) + 1e-8
    frame["y"] = (raw - mean) / std
    frame.attrs.update(target_mean=mean, target_std=std)
    lo, hi = np.quantile(raw[train], [1 / 3, 2 / 3])
    frame["tercile"] = np.digitize(raw, [lo, hi])
    return frame
