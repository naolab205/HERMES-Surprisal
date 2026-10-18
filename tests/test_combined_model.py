"""Material checks for the packaged combined architecture and split statistics."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_charbag_cv import compute_eye_stats
from foundation.hiernet import MultiModalHierSeqNet
from foundation.model import SpatialTransform


def test_eye_standardization_uses_train_only():
    eye = np.zeros((3, 4, 4), dtype=np.float32)
    eye[:, 3, :] = 1
    eye[0, 0, :] = [1, 2, 3, 4]
    eye[1, 0, :] = [3, 4, 5, 6]
    eye[2, 0, :] = 1000  # held-out outlier
    frame = pd.DataFrame({"split": ["train", "train", "test"], "obs_index": [0, 1, 2]})
    slots = [np.array([0]), np.array([1]), np.array([2])]
    train_mean, train_std = compute_eye_stats(eye, slots, frame, "train")
    global_mean, _ = compute_eye_stats(eye, slots, frame, "global")
    assert np.isclose(train_mean[0], 3.5)
    assert train_std[0] > 0
    assert global_mean[0] > 300
    assert train_mean[3] == 0 and train_std[3] == 1


def test_zero_initialized_eye_gate_does_not_change_prediction():
    torch.manual_seed(1)
    spatial = SpatialTransform(np.eye(128, dtype=np.float32)[None], np.ones(1, dtype=np.float32))
    model = MultiModalHierSeqNet(spatial, d_model=16, n_spatial=8,
                                  time_depth=1, char_depth=1, n_heads=2,
                                  max_slots=2, block="modern", dropout=0)
    model.eval()
    eeg = torch.randn(1, 2, 128, 36)
    eye = torch.randn(1, 2, 4, 30)
    mask = torch.ones(1, 2)
    group = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        a = model(eeg, mask, group, eye)["row"]
        b = model(eeg, mask, group, eye * 0)["row"]
    assert torch.allclose(a, b, atol=1e-6)
