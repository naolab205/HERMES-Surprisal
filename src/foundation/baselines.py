"""Published comparison encoders, reimplemented inside this pipeline.

Four architectures from the last three years, all peer reviewed:

* **EEG Conformer** -- Song, Zheng, Liu, Gao, IEEE TNSRE 2023, 31:710-719.
  Convolutional patch embedding followed by a self-attention stack.
* **ATCNet** -- Altaheri, Muhammad, Alsulaiman, IEEE Trans. Industrial
  Informatics 2023, 19(2):2249-2258. Convolutional block, sliding windows, and
  per-window multi-head attention into a dilated temporal convolutional network.
* **MSVTNet** -- IEEE JBHI 2024, 28(12). Multi-scale convolutional branches whose
  tokens are concatenated with a class token and passed to a cross-scale
  transformer, with auxiliary per-branch losses.
* **CTNet** -- Scientific Reports 2024. EEGNet-style embedding into a transformer
  encoder.

Each is used as the per-character encoder and everything downstream is held
identical to the proposed model: the same character-locked epochs, the same
surface Laplacian front-end, the same masked mean pooling over the row's
characters, the same regression head, the same folds, label and optimiser. A
difference in score is then a difference in encoder and nothing else.

These are faithful ports of the published designs rather than the original code:
all four were published for single-trial motor-imagery *classification* over
whole trials, so the classifier head is replaced by this task's scalar readout
and the temporal hyper-parameters are scaled from the sampling rate they assume
to the 128 Hz used here. Any port is an approximation, and that is stated in the
results rather than glossed over.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _BagWrapper(nn.Module):
    """Applies a per-character encoder to a bag and mean-pools over the row."""

    def __init__(self, spatial, encoder: nn.Module, embed_dim: int,
                 dropout: float = 0.25) -> None:
        super().__init__()
        self.spatial_op = spatial
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor,
                group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial_op(eeg, group).reshape(b * s, c, t)
        z = self.encoder(x).reshape(b, s, -1)
        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bs,bsd->bd", weight, z)
        row = self.head(pooled).squeeze(-1)
        return {"row": row, "char": torch.zeros_like(mask),
                "row_from_chars": row, "weights": weight, "embedding": pooled}


class ConformerEncoder(nn.Module):
    """EEG Conformer: convolutional patch embedding then self-attention."""

    def __init__(self, n_channels: int = 128, n_times: int = 166,
                 emb: int = 40, depth: int = 6, heads: int = 10,
                 dropout: float = 0.25) -> None:
        super().__init__()
        # Published kernels assume 250 Hz; scaled to 128 Hz here (25 -> 13,
        # pool 75/15 -> 38/8) so the receptive field in milliseconds matches.
        self.embed = nn.Sequential(
            nn.Conv2d(1, emb, (1, 13), padding=(0, 6)),
            nn.Conv2d(emb, emb, (n_channels, 1)),
            nn.BatchNorm2d(emb), nn.ELU(),
            nn.AvgPool2d((1, 38), stride=(1, 8)),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=2 * emb, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_dim = emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        h = self.transformer(h)
        return h.mean(dim=1)


class _TcnBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad)
        self.drop = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.elu(self.conv1(x)[..., : -self.pad or None]))
        h = self.drop(F.elu(self.conv2(h)[..., : -self.pad or None]))
        return F.elu(x + h)


class AtcnetEncoder(nn.Module):
    """ATCNet: convolutional block, sliding windows, attention, then a TCN."""

    def __init__(self, n_channels: int = 128, n_times: int = 166, f1: int = 16,
                 depth_mult: int = 2, windows: int = 5, tcn_filters: int = 32,
                 dropout: float = 0.3) -> None:
        super().__init__()
        f2 = f1 * depth_mult
        self.conv = nn.Sequential(
            nn.Conv2d(1, f1, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(f2, f2, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 3)), nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(f2, 2, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(f2)
        self.project = nn.Conv1d(f2, tcn_filters, 1)
        self.tcn = nn.Sequential(*[_TcnBlock(tcn_filters, 4, 2 ** i, dropout)
                                   for i in range(2)])
        self.windows = windows
        self.out_dim = tcn_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1)).squeeze(2).transpose(1, 2)   # [N, T', F2]
        # Sliding windows: the published model averages a decision per window,
        # which here becomes an averaged embedding so the shared head applies.
        t = h.shape[1]
        span = max(1, t - self.windows + 1)
        outputs = []
        for start in range(min(self.windows, max(t - span + 1, 1))):
            chunk = h[:, start:start + span]
            attended, _ = self.attn(chunk, chunk, chunk, need_weights=False)
            chunk = self.norm(chunk + attended).transpose(1, 2)
            outputs.append(self.tcn(self.project(chunk))[..., -1])
        return torch.stack(outputs, dim=0).mean(dim=0)


class MsvtnetEncoder(nn.Module):
    """MSVTNet: multi-scale convolutional branches into a cross-scale transformer."""

    def __init__(self, n_channels: int = 128, n_times: int = 166,
                 kernels: tuple[int, ...] = (8, 16, 32, 64), f1: int = 9,
                 depth_mult: int = 2, emb: int = 32, depth: int = 2,
                 heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        f2 = f1 * depth_mult
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, f1, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
                nn.Conv2d(f2, f2, (1, 8), padding=(0, 4), bias=False),
                nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            )
            for k in kernels
        )
        self.project = nn.Linear(f2 * len(kernels), emb)
        self.cls = nn.Parameter(torch.zeros(1, 1, emb))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=2 * emb, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_dim = emb
        # Auxiliary per-branch readouts, as in the published intermediate
        # supervision; exposed so the runner can add them to the loss.
        self.aux = nn.ModuleList(nn.Linear(f2, 1) for _ in kernels)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.unsqueeze(1)
        feats = [branch(x).squeeze(2) for branch in self.branches]   # [N, F2, T_i]
        width = min(f.shape[-1] for f in feats)
        tokens = torch.cat([f[..., :width] for f in feats], dim=1).transpose(1, 2)
        tokens = self.project(tokens)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        out = self.transformer(torch.cat([cls, tokens], dim=1))[:, 0]
        if not return_aux:
            return out
        aux = [head(f.mean(dim=-1)).squeeze(-1) for head, f in zip(self.aux, feats)]
        return out, aux


class CtnetEncoder(nn.Module):
    """CTNet: EEGNet-style embedding into a transformer encoder."""

    def __init__(self, n_channels: int = 128, n_times: int = 166, f1: int = 8,
                 depth_mult: int = 2, emb: int = 16, depth: int = 6,
                 heads: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        f2 = f1 * depth_mult
        self.embed = nn.Sequential(
            nn.Conv2d(1, f1, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, emb, (1, 1), bias=False),
            nn.BatchNorm2d(emb), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.position = nn.Parameter(torch.zeros(1, 64, emb))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=2 * emb, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_dim = emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        h = h + self.position[:, : h.shape[1]]
        return self.transformer(h).mean(dim=1)


class MsvtnetBag(_BagWrapper):
    """MSVTNet wrapper that also surfaces its auxiliary branch predictions."""

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor,
                group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial_op(eeg, group).reshape(b * s, c, t)
        z, aux = self.encoder(x, return_aux=True)
        z = z.reshape(b, s, -1)
        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bs,bsd->bd", weight, z)
        row = self.head(pooled).squeeze(-1)
        aux_rows = [
            torch.einsum("bs,bs->b", weight, a.reshape(b, s)) for a in aux
        ]
        return {"row": row, "char": torch.zeros_like(mask),
                "row_from_chars": row, "weights": weight, "embedding": pooled,
                "aux": aux_rows}


def build_baseline(name: str, spatial, *, n_channels: int = 128,
                   n_times: int = 166, dropout: float = 0.25) -> nn.Module:
    if name == "conformer":
        enc = ConformerEncoder(n_channels, n_times, dropout=dropout)
        return _BagWrapper(spatial, enc, enc.out_dim, dropout)
    if name == "atcnet":
        enc = AtcnetEncoder(n_channels, n_times, dropout=dropout)
        return _BagWrapper(spatial, enc, enc.out_dim, dropout)
    if name == "msvtnet":
        enc = MsvtnetEncoder(n_channels, n_times, dropout=dropout)
        return MsvtnetBag(spatial, enc, enc.out_dim, dropout)
    if name == "ctnet":
        enc = CtnetEncoder(n_channels, n_times, dropout=dropout)
        return _BagWrapper(spatial, enc, enc.out_dim, dropout)
    raise ValueError(f"Unknown baseline: {name}")
