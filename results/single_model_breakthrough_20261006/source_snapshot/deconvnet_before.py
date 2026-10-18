"""Decoder built around schedule-conditioned deconvolution.

Structural difference from the previous models
---------------------------------------------
HierSeqNet pools each character's time axis away before the characters ever meet:
the within-character block produces one vector per slot, and only then does the
cross-character block run. That ordering makes deconvolution impossible, because
separating overlapping responses requires knowing *when* inside each window the
neighbour's response falls, and pooling has already discarded that.

So the time axis is carried through the character axis here. Both are kept live
until after the deconvolution steps, and pooling happens last:

    per-slot time-resolved features   [B, S, T, D]
      -> unrolled schedule deconvolution (subtract shifted neighbours)
      -> within-character sequence block   (over T)
      -> cross-character sequence block    (over S)
      -> masked pooling -> head

The cost is memory: S x T tokens live at once rather than S. With ten slots and
about forty tokens per slot that is four hundred positions, which is ordinary for
attention and is why the two axes are still factorised rather than flattened into
one long sequence.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .deconv import ScheduleDeconvStack
from .hiernet import EyeCharEncoder, SinusoidalTime, _apply, _block
from .modern import ModernStack


class TimeResolvedStem(nn.Module):
    """Spatial filtering and temporal convolution, keeping the time axis."""

    def __init__(self, n_channels: int = 128, n_spatial: int = 64, d_model: int = 96,
                 stride: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, n_spatial, 1, bias=False)
        self.norm0 = nn.BatchNorm1d(n_spatial)
        # ~117 ms depthwise kernel: one N400 complex, so the deconvolution
        # operates on response-shaped components rather than raw samples.
        self.temporal = nn.Conv1d(n_spatial, n_spatial, 15, padding=7,
                                  groups=n_spatial, bias=False)
        self.mix = nn.Conv1d(n_spatial, d_model, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.stride = stride
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.norm0(self.spatial(x)))
        h = F.elu(self.norm1(self.mix(self.temporal(h))))
        return self.drop(F.avg_pool1d(h, self.stride)).transpose(1, 2)


class DeconvNet(nn.Module):
    """Schedule-conditioned deconvolution, then two factorised sequence blocks."""

    def __init__(self, spatial, *, n_channels: int = 128, d_model: int = 96,
                 n_spatial: int = 64, stride: int = 4, time_depth: int = 2,
                 char_depth: int = 2, n_heads: int = 4, max_slots: int = 10,
                 block: str = "modern", dropout: float = 0.1,
                 deconv_steps: int = 2, deconv_radius: int = 2,
                 use_deconv: bool = True, use_eye: bool = False,
                 eye_depth: int = 1, eye_stats: tuple | None = None,
                 char_head: bool = False) -> None:
        super().__init__()
        self.spatial_op = spatial
        self.stem = TimeResolvedStem(n_channels, n_spatial, d_model, stride, dropout)
        self.use_deconv = use_deconv
        self.time_block = _block(block, d_model, n_heads, dropout, time_depth)
        self.time_position = SinusoidalTime(d_model)
        self.char_block = _block(block, d_model, n_heads, dropout, char_depth)
        self.slot_position = nn.Embedding(max_slots, d_model)
        nn.init.normal_(self.slot_position.weight, std=0.02)
        self.use_eye = use_eye
        if use_eye:
            self.eye_encoder = EyeCharEncoder(4, d_model, eye_depth, n_heads,
                                              block, dropout, eye_stats)
            self.eye_gate = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.eye_gate.weight)
            nn.init.zeros_(self.eye_gate.bias)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )
        # Instantiate the candidate-only block after every shared layer. With
        # the same seed, the control and candidate now have identical shared
        # initial weights; the zero gate makes their initial outputs identical.
        if use_deconv:
            self.deconv = ScheduleDeconvStack(d_model, stride, deconv_steps,
                                              deconv_radius, dropout)
        # Auxiliary head is instantiated last, preserving all shared starting
        # weights. It receives neural embeddings only, never local labels.
        self.char_head = nn.Linear(d_model, 1) if char_head else None

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor, group: torch.Tensor,
                eye: torch.Tensor | None = None) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial_op(eeg, group).reshape(b * s, c, t)
        h = self.stem(x)                                   # [B*S, T', D]
        h = h.view(b, s, h.shape[1], h.shape[2])           # [B, S, T', D]

        if self.use_deconv:
            h = self.deconv(h, mask)

        # Within-character, batching slots so attention stays over time only.
        flat = h.reshape(b * s, h.shape[2], h.shape[3])
        if not isinstance(self.time_block, ModernStack):
            flat = self.time_position(flat)
        flat = _apply(self.time_block, flat, None)
        z = torch.cat([flat.mean(dim=1), flat.amax(dim=1)], dim=-1)
        z = z[:, : h.shape[3]] + z[:, h.shape[3]:]         # mean + max, no extra params
        z = z.view(b, s, -1)

        if self.use_eye and eye is not None:
            e = self.eye_encoder(eye.reshape(b * s, eye.shape[2], eye.shape[3]))
            z = z + self.eye_gate(e).view(b, s, -1)

        pad = mask < 0.5
        if bool(pad.all(dim=1).any()):
            pad = pad & ~pad.all(dim=1, keepdim=True)
        slots = torch.arange(s, device=z.device).unsqueeze(0)
        z = _apply(self.char_block, z + self.slot_position(slots), pad)

        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bs,bsd->bd", weight, z)
        row = self.head(pooled).squeeze(-1)
        char = (torch.zeros_like(mask) if self.char_head is None
                else self.char_head(z).squeeze(-1))
        row_from_chars = row if self.char_head is None else (char * weight).sum(dim=1)
        return {"row": row, "char": char,
                "row_from_chars": row_from_chars, "weights": weight, "embedding": pooled}

    @torch.no_grad()
    def deconv_report(self) -> list[float]:
        return self.deconv.gate_magnitudes() if self.use_deconv else []
