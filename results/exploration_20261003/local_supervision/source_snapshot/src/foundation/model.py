"""Character-bag encoder for row-level surprisal decoding.

The row is a bag of character epochs, each on the true physical time axis.  A
shared encoder embeds every character epoch, a masked pooling layer collapses
the bag into a row embedding, and a small head regresses the row's surprisal.

Fixed spatial operators (surface Laplacian, Euclidean Alignment) are applied
inside the model as a per-recording 128x128 matrix.  They are frozen buffers,
not parameters: keeping them here means the character cache stays a single
copy on disk and the transforms cost one matmul on the GPU.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpatialTransform(nn.Module):
    """Per-recording fixed spatial operator plus a per-recording scale.

    ``matrices[g]`` is applied to every epoch of recording ``g``.  It carries
    the surface Laplacian, the Euclidean Alignment whitener, or their product;
    ``scales[g]`` normalizes the result to unit variance using training epochs
    only, so the encoder sees comparable amplitudes across recordings.
    """

    def __init__(self, matrices: np.ndarray, scales: np.ndarray) -> None:
        super().__init__()
        # The amplitude scale is folded into the matrix rather than applied
        # afterwards.  A bare surface Laplacian has large entries, so scaling
        # after the matmul lets the intermediate overflow float16 under autocast
        # and the whole forward pass turns to NaN.
        matrices = np.asarray(matrices, dtype=np.float32) * np.asarray(
            scales, dtype=np.float32
        ).reshape(-1, 1, 1)
        self.register_buffer("matrices", torch.as_tensor(matrices, dtype=torch.float32))

    def forward(self, eeg: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        # eeg [B, S, C, T]; one matrix per item in the batch.  Kept in float32:
        # it is one cheap matmul and the Laplacian's dynamic range is wide.
        with torch.amp.autocast("cuda", enabled=False):
            matrix = self.matrices[group].float()          # [B, C, C]
            return torch.einsum("bcd,bsdt->bsct", matrix, eeg.float())


class CharEncoder(nn.Module):
    """Embeds one character epoch [128 channels x 71 samples] at 128 Hz."""

    def __init__(self, n_channels: int = 128, n_spatial: int = 64, embed_dim: int = 128,
                 dropout: float = 0.25, width: float = 1.0) -> None:
        super().__init__()
        # ``width`` scales every hidden stage together. The step-1 arms used 1.0
        # deliberately, to keep capacity fixed while the representation changed;
        # it is a knob, not a tuned default.
        n_spatial = int(round(n_spatial * width))
        c1 = int(round(96 * width))
        c2 = int(round(128 * width))
        self.spatial = nn.Conv1d(n_channels, n_spatial, kernel_size=1, bias=False)
        self.spatial_norm = nn.BatchNorm1d(n_spatial)
        # ~117 ms kernel at 128 Hz: wide enough to span a P1/N1 complex.
        self.temporal = nn.Conv1d(n_spatial, n_spatial, kernel_size=15, padding=7,
                                  groups=n_spatial, bias=False)
        self.mix = nn.Conv1d(n_spatial, c1, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm1d(c1)
        self.temporal2 = nn.Conv1d(c1, c1, kernel_size=7, padding=3, groups=c1, bias=False)
        self.mix2 = nn.Conv1d(c1, c2, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm1d(c2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.LayerNorm(2 * c2), nn.Linear(2 * c2, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.spatial_norm(self.spatial(x)))
        h = F.elu(self.norm1(self.mix(self.temporal(h))))
        h = F.avg_pool1d(h, 2)
        h = self.drop(h)
        h = F.elu(self.norm2(self.mix2(self.temporal2(h))))
        pooled = torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1)
        return self.proj(pooled)


class MaskedPool(nn.Module):
    """Mean or attention pooling over the character slots of one row."""

    def __init__(self, embed_dim: int, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "attention"}:
            raise ValueError(f"Unknown pooling mode: {mode}")
        self.mode = mode
        if mode == "attention":
            self.score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.mode == "attention":
            logits = self.score(z).squeeze(-1)
            logits = logits.masked_fill(mask < 0.5, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1) * (mask > 0.5)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bs,bsd->bd", weights, z)
        return pooled, weights


class CharBagModel(nn.Module):
    def __init__(
        self,
        spatial: SpatialTransform,
        *,
        n_channels: int = 128,
        embed_dim: int = 128,
        pooling: str = "mean",
        dropout: float = 0.25,
        char_head: bool = False,
        width: float = 1.0,
    ) -> None:
        super().__init__()
        self.spatial = spatial
        self.encoder = CharEncoder(n_channels=n_channels, embed_dim=embed_dim,
                                   dropout=dropout, width=width)
        self.pool = MaskedPool(embed_dim, mode=pooling)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )
        if char_head:
            # Its own head: this one predicts a character's surprisal, a
            # different quantity from the pooled row readout.
            self.char_head = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(64, 1),
            )
        else:
            self.char_head = self.head

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor, group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial(eeg, group)
        z = self.encoder(x.reshape(b * s, c, t)).reshape(b, s, -1)
        pooled, weights = self.pool(z, mask)
        char = self.char_head(z).squeeze(-1)
        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "row": self.head(pooled).squeeze(-1),
            "char": char,
            "row_from_chars": (char * weight).sum(dim=1),
            "weights": weights,
            "embedding": pooled,
        }


def sum_constrained_loss(
    char_pred: torch.Tensor,
    mask: torch.Tensor,
    char_target: torch.Tensor,
    row_target: torch.Tensor,
    *,
    lambda_instance: float = 1.0,
    lambda_consistency: float = 1.0,
    char_scale: float = 1.0,
    supervision_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local contribution regression plus masked-mean row consistency.

    Targets must be converted to slot contribution units by load_char_targets;
    they are not unmodified token surprisals. Incomplete bags retain the
    external row loss but are excluded from these two auxiliary terms.
    """
    weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    row_from_chars = (char_pred * weights).sum(dim=1)
    # Characters must be standardized by the ROW label's statistics to preserve
    # the sum identity, which leaves their spread about 3x the row's (measured:
    # 3.026 vs 1.005 nats) and their squared error about 9x. Dividing by the
    # training character scale puts both terms at O(1) so lambda means what it
    # looks like; the targets themselves are untouched, so the identity holds.
    scale = max(float(char_scale), 1e-6) ** 2
    valid = (torch.ones_like(row_target) if supervision_valid is None
             else supervision_valid.to(row_target.dtype))
    supervised_mask = mask * valid.unsqueeze(1)
    instance = (((char_pred - char_target) ** 2) * supervised_mask).sum() / supervised_mask.sum().clamp_min(1.0)
    consistency = (((row_from_chars - row_target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
    loss = lambda_instance * (instance / scale) + lambda_consistency * consistency
    return loss, row_from_chars


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson r inside the batch; complements MSE on a ranking endpoint."""
    if pred.numel() < 3:
        return pred.new_zeros(())
    p = pred - pred.mean()
    t = target - target.mean()
    denom = (p.norm() * t.norm()).clamp_min(1e-8)
    return 1.0 - (p @ t) / denom
