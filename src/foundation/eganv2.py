"""EGAN with a sequence backbone in place of the convolutional waveform branch.

Why revisit components a full ablation rejected
----------------------------------------------
The first EGAN ablation put the learnable-xDAWN geometry branch at -0.013 item
rho and the precision pooling at -0.005, and both were dropped. That ablation
ran with a purely convolutional waveform branch, and a main effect measured
against one backbone does not transfer to another: the most likely reason the
geometry branch cost accuracy is that it was *redundant* with the convolution,
not that covariance geometry is uninformative. A branch that localises events in
time (sequence) and a branch that reads channel covariance (geometry) have less
overlap than two variants of the same convolution, so the interaction is worth
re-measuring rather than assumed.

The sequence backbone itself is now established rather than hoped for: the
hierarchical two-level model reached item rho 0.3835 against the convolutional
0.3798, with 7/7 folds improved on all three endpoints against the frozen
baseline and 94.7% retention under nuisance control.

Retained from the original design, each with the measurement behind it
--------------------------------------------------------------------
* learnable xDAWN into an SPD manifold stage -- fixed prototypes lifted a plain
  Riemannian decoder from 0.080 to 0.280, so the evoked cross-term is what makes
  a covariance feature see a phase-locked response;
* FiLM coupling rather than concatenation -- geometry and waveform predictions
  correlate 0.47, complementary without being redundant;
* precision pooling rather than softmax attention -- attention pooling lost to
  the mean, but inverse-variance weighting asks about reliability, not salience;
* rank-aware whitening front-end -- the covariance is rank deficient by ~5
  dimensions (condition number 1.9e9) from bad-channel interpolation;
* narrow, not wide -- widening moved trial rho from 0.212 down to 0.190.

Two lessons already paid for are built in: the super-trial is row-normalised (an
ill-conditioned covariance made eigh fail outright), and pooling happens in
representation space (pooling scalar readouts collapsed the output variance to
0.05 against a unit-variance target).
"""
from __future__ import annotations

import torch
from torch import nn

from .egan import GeometryBranch, PrecisionPool
from .hiernet import CharTimeEncoder, _apply, _block


class EganSeq(nn.Module):
    """Sequence waveform branch, geometry branch, FiLM, cross-character block, precision pool."""

    def __init__(self, spatial, *, n_channels: int = 128, n_times: int = 115,
                 d_model: int = 96, n_spatial: int = 64, stride: int = 4,
                 time_depth: int = 2, char_depth: int = 2, n_heads: int = 4,
                 max_slots: int = 10, block: str = "transformer",
                 dropout: float = 0.1, n_filters: int = 12, n_prototypes: int = 6,
                 n_manifold: int = 12, use_geometry: bool = True,
                 couple: bool = True, pool_mode: str = "precision",
                 use_char_block: bool = True) -> None:
        super().__init__()
        self.spatial_op = spatial
        self.use_geometry = use_geometry
        self.use_char_block = use_char_block
        self.pool_mode = pool_mode

        self.waveform = CharTimeEncoder(
            n_channels, n_spatial, d_model, stride, time_depth, n_heads, block, dropout
        )
        self.geometry = (
            GeometryBranch(n_channels, n_times, n_filters, n_prototypes,
                           n_manifold, d_model)
            if use_geometry else None
        )
        self.couple = couple and use_geometry
        if self.couple:
            # Zero-initialised, so training starts at the waveform branch alone
            # and any coupling has to be earned.
            self.film = nn.Linear(d_model, 2 * d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        fuse_in = 2 * d_model if use_geometry else d_model
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in), nn.Linear(fuse_in, d_model), nn.GELU(),
            nn.Dropout(dropout),
        )
        if use_char_block:
            self.slot_position = nn.Embedding(max_slots, d_model)
            nn.init.normal_(self.slot_position.weight, std=0.02)
            self.char_block = _block(block, d_model, n_heads, dropout, char_depth)
        self.pool = PrecisionPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor,
                group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial_op(eeg, group).reshape(b * s, c, t)
        wave = self.waveform(x)
        if self.use_geometry:
            # eigh is unstable in half precision; the manifold stage stays float32.
            with torch.amp.autocast("cuda", enabled=False):
                geom = self.geometry(x.float())
            if self.couple:
                gamma, beta = self.film(geom).chunk(2, dim=-1)
                wave = wave * (1 + gamma) + beta
            merged = torch.cat([wave, geom], dim=-1)
        else:
            merged = wave
        z = self.fuse(merged).reshape(b, s, -1)
        if self.use_char_block:
            slots = torch.arange(s, device=z.device).unsqueeze(0)
            z = z + self.slot_position(slots)
            pad = mask < 0.5
            if bool(pad.all(dim=1).any()):
                pad = pad & ~pad.all(dim=1, keepdim=True)
            z = _apply(self.char_block, z, pad)
        if self.pool_mode == "mean":
            weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled = torch.einsum("bs,bsd->bd", weight, z)
        else:
            pooled, weight = self.pool(z, mask)
        row = self.head(pooled).squeeze(-1)
        return {"row": row, "char": torch.zeros_like(mask),
                "row_from_chars": row, "weights": weight, "embedding": pooled}
