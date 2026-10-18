#!/usr/bin/env python3
"""Verify punctuation conservation, physical-slot alignment and gradient masks."""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import foundation.data as data  # noqa: E402
from foundation.deconvnet import DeconvNet  # noqa: E402
from foundation.model import SpatialTransform, sum_constrained_loss  # noqa: E402


def main() -> None:
    with TemporaryDirectory() as directory:
        csv = Path(directory) / "characters.csv"
        pd.DataFrame({"book": ["B", "B"], "run": [1, 1], "row_idx": [1, 1],
                      "char_slot": [0, 1], "char_surprisal": [1., 5.],
                      "char": ["甲", "乙"],
                      "n_tokens": [3, 3]}).to_csv(csv, index=False)
        old = data.CHAR_SURPRISAL_CSV
        data.CHAR_SURPRISAL_CSV = csv
        try:
            frame = pd.DataFrame({"book": ["B", "B"], "run": [1, 1],
                                  "row_idx": [1, 1], "n_slots": [2, 1],
                                  "obs_index": [0, 1], "surprisal": [2., 2.],
                                  "text": ["甲，乙", "甲，乙"],
                                  "y": [.5, .5]})
            frame.attrs.update(target_mean=1., target_std=2.)
            target, valid = data.load_char_targets(
                frame, [np.array([0, 1]), np.array([2])], frame, max_slots=2,
                physical_slots=np.array([0, 1, 1]), return_validity=True)
            assert np.allclose(target[0].mean(), .5)
            assert np.allclose(target[1, 0], target[0, 1])
            assert np.array_equal(valid, [1., 0.])
            # A truncated tokenizer string cannot masquerade as a physical
            # highlighted character even if its slot count happens to match.
            broken = pd.read_csv(csv)
            broken.loc[0, "char"] = "["
            broken.to_csv(csv, index=False)
            bad_target, bad_valid = data.load_char_targets(
                frame, [np.array([0, 1]), np.array([2])], frame, max_slots=2,
                physical_slots=np.array([0, 1, 1]), return_validity=True)
            assert np.array_equal(bad_valid, [0., 0.])
            assert np.array_equal(bad_target, np.zeros_like(bad_target))
        finally:
            data.CHAR_SURPRISAL_CSV = old
    pred = torch.tensor(target.copy(), requires_grad=True)
    with torch.no_grad():
        pred[1, 0] = 100.
    loss, _ = sum_constrained_loss(pred, torch.tensor([[1., 1.], [1., 0.]]),
                                  torch.tensor(target), torch.tensor([.5, .5]),
                                  supervision_valid=torch.tensor(valid))
    loss.backward()
    assert abs(float(loss.detach())) < 1e-8 and pred.grad[1].abs().sum() == 0

    def build(head):
        torch.manual_seed(71)
        spatial = SpatialTransform(np.eye(8, dtype=np.float32)[None],
                                   np.ones(1, dtype=np.float32))
        return DeconvNet(spatial, n_channels=8, n_spatial=8, d_model=16,
                         time_depth=1, char_depth=1, max_slots=3,
                         use_deconv=False, dropout=0., char_head=head)
    ordinary, local = build(False), build(True)
    ordinary.eval(); local.eval()
    eeg = torch.randn(2, 3, 8, 40)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    group = torch.zeros(2, dtype=torch.long)
    out = local(eeg, mask, group)
    assert torch.equal(ordinary(eeg, mask, group)["row"], out["row"])
    out["char"].square().mean().backward()
    assert local.char_head.weight.grad.abs().sum() > 0
    assert local.stem.spatial.weight.grad.abs().sum() > 0
    print("verified: token/slot conservation, actual-slot mapping, incomplete-bag masks, "
          "identical initial row output and nonzero neural/local-head gradients")


if __name__ == "__main__":
    main()
