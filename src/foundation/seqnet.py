"""Sequence models over the whole reading row.

Motivation, from a measurement rather than from architecture fashion
-------------------------------------------------------------------
Characters arrive every ~420 ms and the surprisal response estimated by the
temporal response function runs to ~790 ms, so each character's response
overlaps the next two. Extending the character window from 450 to 800 ms
improved all seven folds by 22% in item rho, which proves the overlapping
interval carries signal. But a bag of character epochs followed by mean pooling
*cannot separate* that overlap: every epoch holds a mixture, and pooling
averages mixtures. A TRF deconvolves it linearly; a sequence model over the
row's own continuous signal can in principle learn a nonlinear version.

That is a narrower claim than "CNNs have a limited receptive field". It is also
different from the attention-pooling ablation that failed here: that asked which
of ten characters matters most, and at a 1:70 single-trial signal-to-noise ratio
none does. This asks how overlapping responses combine over time.

Reference points: EEG Conformer (Song et al., IEEE TNSRE 2023, 31:710-719)
pairs a convolutional stem with self-attention for exactly this local/global
split; MI-Mamba (Ann. N.Y. Acad. Sci. 2025) uses convolution for space and a
state-space block for global time at a sixth of the parameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SpatialStem(nn.Module):
    """Fixed spatial operator, then learned spatial filters and a strided stem.

    The stem downsamples time by ``stride`` so the sequence handed to the global
    block is a few hundred steps rather than a few thousand, which is what makes
    attention affordable at full-row length.
    """

    def __init__(self, n_channels: int = 128, n_spatial: int = 64,
                 d_model: int = 128, stride: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, n_spatial, 1, bias=False)
        self.norm0 = nn.BatchNorm1d(n_spatial)
        # ~117 ms depthwise kernel: one N400 complex before any downsampling.
        self.temporal = nn.Conv1d(n_spatial, n_spatial, 15, padding=7,
                                  groups=n_spatial, bias=False)
        self.mix = nn.Conv1d(n_spatial, d_model, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.stride = stride
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.norm0(self.spatial(x)))
        h = F.elu(self.norm1(self.mix(self.temporal(h))))
        h = F.avg_pool1d(h, self.stride)
        return self.drop(h).transpose(1, 2)          # [B, T', d_model]


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class MinimalSSM(nn.Module):
    """Diagonal selective state-space block in pure PyTorch.

    A stand-in for Mamba when its CUDA kernel is unavailable. The recurrence is
    diagonal, so it runs as a parallel scan in log time via cumulative sums in
    log space; gate and step size are input dependent, which is the selective
    part that distinguishes this from a fixed linear filter.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, 4, padding=3, groups=d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.b_proj = nn.Linear(d_inner, d_state)
        self.c_proj = nn.Linear(d_inner, d_state)
        # Negative real log-spaced decay rates, as in S4/S6 initialisation.
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.d_state = d_state
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :t].transpose(1, 2)
        u = F.silu(u)
        dt = F.softplus(self.dt_proj(u))                       # [B, T, d_inner]
        a = -torch.exp(self.a_log)                             # [d_state]
        b_t = self.b_proj(u)
        c_t = self.c_proj(u)
        # Diagonal recurrence h_t = exp(dt*a) h_{t-1} + dt*B_t*u_t.
        # The closed-form parallel scan -- divide by exp(cumulative log decay),
        # cumulative-sum, multiply back -- overflows: the cumulative decay grows
        # without bound over a few hundred steps, so its reciprocal reaches inf
        # and every prediction becomes NaN. A sequential scan over the ~165
        # tokens the stem produces is stable and fast enough on GPU.
        decay = torch.exp(torch.einsum("btd,s->btds", dt, a))
        drive = torch.einsum("btd,bts->btds", dt * u, b_t)
        state = torch.zeros_like(drive[:, 0])
        outputs = []
        for step in range(x.shape[1]):
            state = decay[:, step] * state + drive[:, step]
            outputs.append(torch.einsum("bds,bs->bd", state, c_t[:, step]))
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y * F.silu(gate))


class RowSequenceNet(nn.Module):
    """Spatial stem, a global temporal block, and a masked readout."""

    def __init__(self, spatial, *, n_channels: int = 128, d_model: int = 128,
                 n_spatial: int = 64, stride: int = 4, depth: int = 2,
                 n_heads: int = 4, block: str = "transformer",
                 dropout: float = 0.1) -> None:
        super().__init__()
        if block not in {"transformer", "ssm"}:
            raise ValueError(f"Unknown global block: {block}")
        self.spatial_op = spatial
        self.stem = SpatialStem(n_channels, n_spatial, d_model, stride, dropout)
        self.block_kind = block
        if block == "transformer":
            self.position = SinusoidalPosition(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True,
            )
            self.global_block = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(depth))
            self.blocks = nn.ModuleList(MinimalSSM(d_model) for _ in range(depth))
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.stride = stride

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor,
                group: torch.Tensor) -> dict:
        # eeg [B, 1, C, T] so the same SpatialTransform buffer applies unchanged.
        x = self.spatial_op(eeg, group).squeeze(1)
        h = self.stem(x)
        # Sample mask -> token mask, matching the stem's average pooling.
        token_mask = F.avg_pool1d(mask.unsqueeze(1), self.stride).squeeze(1)
        token_mask = (token_mask > 0.5).float()[:, : h.shape[1]]
        if token_mask.sum() == 0:
            token_mask = torch.ones_like(token_mask)
        if self.block_kind == "transformer":
            h = self.global_block(self.position(h),
                                 src_key_padding_mask=(token_mask < 0.5))
        else:
            for norm, blk in zip(self.norms, self.blocks):
                h = h + blk(norm(h)) * token_mask.unsqueeze(-1)
        weight = token_mask / token_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mean = torch.einsum("bt,btd->bd", weight, h)
        peak = h.masked_fill(token_mask.unsqueeze(-1) < 0.5, float("-inf")).amax(dim=1)
        pooled = torch.cat([mean, torch.nan_to_num(peak, neginf=0.0)], dim=-1)
        row = self.head(pooled).squeeze(-1)
        return {"row": row, "char": torch.zeros_like(mask[:, :1]),
                "row_from_chars": row, "weights": weight, "embedding": pooled}
