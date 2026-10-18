"""Safety checks for the task-specific EEG overlap module."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foundation.hiernet import (  # noqa: E402
    CenteredEvidencePool, MultiModalHierSeqNet, OverlapDemixer,
)
from foundation.model import SpatialTransform  # noqa: E402


def test_zero_initialization_preserves_eeg_only_model():
    spatial = SpatialTransform(np.eye(128, dtype=np.float32)[None],
                               np.ones(1, dtype=np.float32))
    kwargs = dict(d_model=16, n_spatial=8, time_depth=1, char_depth=1,
                  n_heads=2, max_slots=3, block="modern", dropout=0,
                  use_eye=False)
    torch.manual_seed(101)
    baseline = MultiModalHierSeqNet(spatial, **kwargs).eval()
    torch.manual_seed(101)
    candidate = MultiModalHierSeqNet(spatial, overlap_demix=True, **kwargs).eval()
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    eeg = torch.randn(2, 3, 128, 167)
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    group = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        a = baseline(eeg, mask, group)["row"]
        b = candidate(eeg, mask, group)["row"]
    assert torch.allclose(a, b, atol=1e-6)


def test_padded_slots_cannot_change_valid_overlap_correction():
    module = OverlapDemixer(8)
    with torch.no_grad():
        module.middle_projection.weight.fill_(0.01)
        module.late_projection.weight.fill_(0.01)
    embedding = torch.randn(1, 3, 8)
    local = torch.randn(1, 3, 41, 8)
    mask = torch.tensor([[1., 1., 0.]])
    changed = local.clone()
    changed[:, 2] = torch.randn_like(changed[:, 2]) * 1000
    a = module(embedding, local, mask)
    b = module(embedding, changed, mask)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6)


def test_overlap_residual_receives_gradient():
    module = OverlapDemixer(8)
    embedding = torch.zeros(1, 3, 8)
    local = torch.randn(1, 3, 41, 8)
    mask = torch.ones(1, 3)
    # A linear readout exposes whether the zero-initialized projections can
    # learn on the first optimization step.
    module(embedding, local, mask).sum().backward()
    assert module.middle_projection.weight.grad is not None
    assert module.middle_projection.weight.grad.abs().sum() > 0


def test_centered_evidence_is_count_safe_and_initially_identical():
    module = CenteredEvidencePool(8)
    pooled = torch.randn(2, 8)
    characters = torch.randn(2, 4, 8)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]])
    assert torch.allclose(module(pooled, characters, mask), pooled)
    with torch.no_grad():
        module.project.weight.fill_(0.01)
    baseline = module(pooled, characters, mask)
    shifted = characters + torch.randn(2, 1, 8) * 50
    assert torch.allclose(module(pooled, shifted, mask), baseline, atol=1e-5)
    changed_padding = characters.clone()
    changed_padding[mask == 0] = 1000
    assert torch.allclose(module(pooled, changed_padding, mask), baseline, atol=1e-5)


def test_centered_evidence_full_model_matches_initial_baseline():
    spatial = SpatialTransform(np.eye(128, dtype=np.float32)[None],
                               np.ones(1, dtype=np.float32))
    kwargs = dict(d_model=16, n_spatial=8, time_depth=1, char_depth=1,
                  n_heads=2, max_slots=3, block="modern", dropout=0,
                  use_eye=False)
    torch.manual_seed(102)
    baseline = MultiModalHierSeqNet(spatial, **kwargs).eval()
    torch.manual_seed(102)
    candidate = MultiModalHierSeqNet(spatial, centered_evidence=True, **kwargs).eval()
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    eeg = torch.randn(2, 3, 128, 167)
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    group = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(baseline(eeg, mask, group)["row"],
                              candidate(eeg, mask, group)["row"], atol=1e-6)
