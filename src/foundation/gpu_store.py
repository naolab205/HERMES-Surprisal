"""GPU-resident epoch store.

The character cache is 917,123 epochs of [128 channels x 71 samples] in float16
-- about 16.6 GB, which fits in one 49 GB RTX 4090 with room to spare. Keeping
it on the device removes the host dataloader entirely: batches become an index
gather rather than a random read from a memory map, a float32 cast, a pad and a
host-to-device copy.

This is a throughput change only. The sampling order, batch size and batch
contents are identical to the dataloader path, so results are unaffected.
"""
from __future__ import annotations

import numpy as np
import torch


class GpuEpochStore:
    """All epochs plus the bag index, resident on one device."""

    def __init__(
        self,
        shards,
        slot_index: list[np.ndarray],
        *,
        device: torch.device,
        max_slots: int = 10,
        eye: np.ndarray | None = None,
    ) -> None:
        self.device = device
        self.max_slots = int(max_slots)

        total = len(shards)
        first = shards.shards[0]
        self.data = torch.empty(
            (total,) + tuple(first.shape[1:]), dtype=torch.float16, device=device
        )
        offset = 0
        # Copy shard by shard, in host-sized chunks, so neither the host nor the
        # device ever holds a second full copy.
        for shard in shards.shards:
            n = len(shard)
            step = 20000
            for lo in range(0, n, step):
                hi = min(lo + step, n)
                block = torch.from_numpy(np.ascontiguousarray(shard[lo:hi]))
                self.data[offset + lo : offset + hi].copy_(block.to(device, non_blocking=True))
            offset += n

        n_obs = len(slot_index)
        gather = np.zeros((n_obs, self.max_slots), dtype=np.int64)
        mask = np.zeros((n_obs, self.max_slots), dtype=np.float32)
        for i, indices in enumerate(slot_index):
            take = min(len(indices), self.max_slots)
            gather[i, :take] = indices[:take]
            mask[i, :take] = 1.0
        self.gather = torch.as_tensor(gather, device=device)
        self.mask = torch.as_tensor(mask, device=device)
        # The eye tensor is indexed by the same global epoch id, so one gather
        # serves both modalities and they cannot drift apart.
        self.eye = (
            None if eye is None
            else torch.as_tensor(np.ascontiguousarray(eye), dtype=torch.float16,
                                 device=device)
        )
        self.eye_clean = None

    @property
    def bytes_on_device(self) -> int:
        return self.data.numel() * self.data.element_size()

    def batch(self, obs_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B, slots, channels, time]`` float32 and its ``[B, slots]`` mask.

        Padding slots gather row 0 and are then zeroed by the mask, which keeps
        the gather rectangular without a scatter.
        """
        rows = self.gather.index_select(0, obs_indices)          # [B, S]
        mask = self.mask.index_select(0, obs_indices)            # [B, S]
        flat = self.data.index_select(0, rows.reshape(-1)).float()
        epochs = flat.view(rows.shape[0], self.max_slots, *self.data.shape[1:])
        epochs = epochs * mask.view(*mask.shape, 1, 1)
        if self.eye is None:
            return epochs, mask
        eye_flat = self.eye.index_select(0, rows.reshape(-1)).float()
        eye = eye_flat.view(rows.shape[0], self.max_slots, *self.eye.shape[1:])
        return epochs, mask, eye * mask.view(*mask.shape, 1, 1)


class GpuBatcher:
    """Deterministic shuffled batching over a split, entirely on device."""

    def __init__(
        self,
        store: GpuEpochStore,
        obs_positions: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray,
        *,
        book_ids: np.ndarray | None = None,
        item_ids: np.ndarray | None = None,
        paired_items: bool = False,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        char_targets: np.ndarray | None = None,
        char_valid: np.ndarray | None = None,
        time_lengths: np.ndarray | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.store = store
        device = store.device
        self.positions = torch.as_tensor(np.asarray(obs_positions), device=device)
        self.y = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
        self.group = torch.as_tensor(np.asarray(groups, dtype=np.int64), device=device)
        self.book = (
            None if book_ids is None
            else torch.as_tensor(np.asarray(book_ids, dtype=np.int64), device=device)
        )
        self.item = (
            None if item_ids is None
            else torch.as_tensor(np.asarray(item_ids, dtype=np.int64), device=device)
        )
        self.paired_items = bool(paired_items)
        if self.paired_items:
            if self.item is None or not shuffle or batch_size % 2:
                raise ValueError("Paired-item batches require item IDs, shuffle, and even batch size")
            _, inverse = np.unique(np.asarray(item_ids), return_inverse=True)
            members = [np.flatnonzero(inverse == item) for item in range(inverse.max() + 1)]
            if min(map(len, members)) < 2:
                raise ValueError("Every training item needs at least two distinct readings")
            width = max(map(len, members))
            table = np.zeros((len(members), width), dtype=np.int64)
            for item, rows in enumerate(members):
                table[item, :len(rows)] = rows
            self.pair_table = torch.as_tensor(table, device=device)
            self.pair_counts = torch.as_tensor([len(rows) for rows in members],
                                               device=device)
        self.y_char = (
            None if char_targets is None
            else torch.as_tensor(np.asarray(char_targets, dtype=np.float32), device=device)
        )
        self.char_valid = (None if char_valid is None else
                           torch.as_tensor(np.asarray(char_valid, dtype=np.float32), device=device))
        # For row-sequence inputs the mask marks valid TIME steps, not slots:
        # every row is one padded sequence and the model needs to know where the
        # padding starts.
        self.time_lengths = (
            None if time_lengths is None
            else torch.as_tensor(np.asarray(time_lengths, dtype=np.int64), device=device)
        )
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator

    def __len__(self) -> int:
        n = len(self.positions)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def n_samples(self) -> int:
        return len(self.positions)

    def __iter__(self):
        n = len(self.positions)
        if self.paired_items:
            # Each step draws distinct readers of the same training item in
            # adjacent positions. Uniform item sampling matches the item-level
            # endpoint; no validation/test item is in this table.
            selections = []
            for _ in range(len(self)):
                chosen = torch.randint(len(self.pair_counts),
                                       (self.batch_size // 2,), device=self.positions.device,
                                       generator=self.generator)
                counts = self.pair_counts[chosen]
                first = (torch.rand(len(chosen), device=self.positions.device,
                                    generator=self.generator) * counts).long()
                second = (torch.rand(len(chosen), device=self.positions.device,
                                     generator=self.generator) * (counts - 1)).long()
                second = second + (second >= first).long()
                pair = torch.stack((self.pair_table[chosen, first],
                                    self.pair_table[chosen, second]), dim=1)
                selections.append(pair.reshape(-1))
        else:
            order = (
                torch.randperm(n, device=self.positions.device, generator=self.generator)
                if self.shuffle
                else torch.arange(n, device=self.positions.device)
            )
            limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
            selections = (order[lo:lo + self.batch_size]
                          for lo in range(0, limit, self.batch_size))
        for sel in selections:
            picked = self.store.batch(self.positions.index_select(0, sel))
            eye_batch = None
            if len(picked) == 3:
                eeg, mask, eye_batch = picked
            else:
                eeg, mask = picked
            group = self.group.index_select(0, sel)
            if self.store.eye_clean is not None:
                eeg = self.store.eye_clean.apply(eeg, eye_batch, group)
            if self.time_lengths is not None:
                width = eeg.shape[-1]
                index = torch.arange(width, device=eeg.device).unsqueeze(0)
                mask = (index < self.time_lengths.index_select(0, sel).unsqueeze(1)).float()
            batch = {
                "eeg": eeg,
                "mask": mask,
                "y": self.y.index_select(0, sel),
                "group": group,
                "index": sel,
            }
            if self.book is not None:
                batch["book"] = self.book.index_select(0, sel)
            if self.item is not None:
                batch["item"] = self.item.index_select(0, sel)
                batch["obs_position"] = self.positions.index_select(0, sel)
            if self.y_char is not None:
                batch["y_char"] = self.y_char.index_select(0, sel)
            if self.char_valid is not None:
                batch["char_supervision_valid"] = self.char_valid.index_select(0, sel)
            if eye_batch is not None:
                batch["eye"] = eye_batch
            yield batch
