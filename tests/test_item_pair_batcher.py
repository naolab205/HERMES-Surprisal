"""Training-only repeated-reader batch checks."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foundation.gpu_store import GpuBatcher  # noqa: E402


class TinyStore:
    device = torch.device("cpu")
    eye_clean = None

    def batch(self, indices):
        return indices.float().reshape(-1, 1, 1, 1), torch.ones(len(indices), 1)


def test_item_pair_batcher_selects_distinct_readers_of_same_item():
    item_ids = np.repeat(np.arange(4), 3)
    batcher = GpuBatcher(
        TinyStore(), np.arange(12), item_ids.astype(np.float32),
        np.zeros(12, dtype=np.int64), item_ids=item_ids,
        paired_items=True, batch_size=4, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(23),
    )
    for batch in batcher:
        assert torch.equal(batch["item"][0::2], batch["item"][1::2])
        assert torch.all(batch["obs_position"][0::2] != batch["obs_position"][1::2])
        assert torch.equal(batch["y"][0::2], batch["y"][1::2])


def test_item_pair_batcher_rejects_single_reader_item():
    try:
        GpuBatcher(
            TinyStore(), np.arange(3), np.zeros(3), np.zeros(3, dtype=np.int64),
            item_ids=np.array([0, 0, 1]), paired_items=True,
            batch_size=2, shuffle=True, drop_last=True,
        )
    except ValueError as error:
        assert "two distinct readings" in str(error)
    else:
        raise AssertionError("Singleton item was accepted")
