"""Evoked-Geometry Accumulation Network.

Every component below is forced by a measurement made on this corpus rather
than chosen by preference. The measurements, and what each one rules out:

1.  A plain Riemannian tangent-space decoder reaches item rho 0.080. Adding
    xDAWN evoked prototypes to the super-trial raises it to 0.280 -- a factor of
    3.5. Second-order statistics are phase-blind, so a covariance branch only
    sees a time-locked response through its cross-terms with a template. The
    geometry branch here therefore keeps learnable prototypes, trained
    end-to-end instead of fixed to class averages.

2.  The surprisal temporal response function peaks at 0.15 microvolts against
    roughly 10 microvolts of single-trial noise, and per-character surprisal is
    undecodable (within-row rho 0.003 over 130,000 characters). Evidence has to
    be accumulated across the row, and no per-character decision is reliable.

3.  Predictions of the waveform CNN and of the Riemannian arm correlate at
    rho 0.47, not 0.9. The two views are complementary, so they are coupled
    inside the model rather than averaged after it.

4.  Widening the encoder moves trial rho from 0.212 down to 0.190. Capacity is
    not the constraint; inductive bias is. Nothing here is wide.

5.  Softmax attention pooling lost to plain mean pooling on all three
    endpoints. At this signal-to-noise ratio there is no salient character to
    select. Inverse-variance weighting is a different claim -- it weights by
    estimated reliability, is non-competitive across slots, and is the
    statistically correct way to combine noisy independent estimates.

6.  The spatial covariance of this data tier is rank deficient by roughly five
    dimensions (condition number 1.9e9) because bad channels were interpolated
    and the reference is the average. Whitening is rank-aware or it amplifies
    empty directions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LearnableXdawn(nn.Module):
    """Super-trial construction with learnable spatial filters and prototypes.

    xDAWN builds ``[P ; W X]`` and takes its covariance, so the off-diagonal
    block holds ``P (W X)^T`` -- the projection of the trial onto the evoked
    template. That cross-term is the only route by which a covariance feature
    can see a phase-locked response, which is why the fixed-prototype version
    tripled the plain covariance score. Here ``W`` and ``P`` are parameters, so
    the template is fitted to the endpoint rather than to a tercile split.
    """

    def __init__(self, n_channels: int, n_filters: int, n_prototypes: int,
                 n_times: int) -> None:
        super().__init__()
        self.spatial = nn.Parameter(torch.empty(n_filters, n_channels))
        nn.init.orthogonal_(self.spatial)
        # Prototypes must start at the scale of the filtered signal. Measured on
        # this cache with unit-variance input: the filtered block of the
        # super-trial covariance sits at 2.1e-1, while 0.01-scale prototypes put
        # the cross-terms -- the only place evoked evidence lives -- at 7.3e-4,
        # which is 27x below the shrinkage floor of 2.0e-2. The branch then
        # receives no gradient and the whole model fails to train. Unit variance
        # per prototype puts the cross-terms on par with the filtered block.
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_times))
        self.width = n_filters + n_prototypes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, channels, time] -> [N, filters + prototypes, time]
        filtered = torch.einsum("fc,nct->nft", self.spatial, x)
        prototypes = self.prototypes.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([filtered, prototypes], dim=1)


class SpdCovariance(nn.Module):
    """Shrunk spatial covariance of a super-trial, guaranteed positive definite."""

    def __init__(self, shrinkage: float = 0.02) -> None:
        super().__init__()
        self.shrinkage = float(shrinkage)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal - signal.mean(dim=-1, keepdim=True)
        # Normalise each row of the super-trial to unit RMS, making this a
        # correlation matrix. Without it the learnable prototypes and the
        # spatially filtered channels sit on scales two orders apart, the
        # covariance becomes ill conditioned, and torch.linalg.eigh either
        # raises "too many repeated eigenvalues" or returns gradients that are
        # numerically meaningless -- which is what silently stopped this branch
        # from training. It is also the right quantity on its own terms: the
        # discriminative content of an xDAWN cross-term is the normalised
        # correlation between trial and template, not its raw amplitude.
        signal = signal / signal.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        cov = signal @ signal.transpose(-1, -2) / signal.shape[-1]
        width = cov.shape[-1]
        eye = torch.eye(width, device=cov.device, dtype=cov.dtype)
        return (1 - self.shrinkage) * cov + self.shrinkage * eye


class BiMapLogEig(nn.Module):
    """One SPDNet stage: congruence projection, eigenvalue floor, matrix log.

    ``BiMap`` moves the covariance to a smaller manifold, ``ReEig`` floors the
    spectrum so the log stays finite, and ``LogEig`` maps to the tangent space
    at the identity where the rest of the network can treat it as a vector
    (Huang and Van Gool, AAAI 2017).
    """

    def __init__(self, n_in: int, n_out: int, eps: float = 1e-4) -> None:
        super().__init__()
        weight = torch.empty(n_in, n_out)
        nn.init.orthogonal_(weight)
        self.weight = nn.Parameter(weight)
        self.eps = float(eps)
        self.n_out = n_out
        index = torch.triu_indices(n_out, n_out)
        self.register_buffer("triu", index)
        scale = torch.where(index[0] == index[1], 1.0, float(np.sqrt(2.0)))
        self.register_buffer("triu_scale", scale)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        # Semi-orthogonalise so the congruence stays well conditioned.
        q, _ = torch.linalg.qr(self.weight)
        projected = q.transpose(0, 1) @ cov @ q
        projected = 0.5 * (projected + projected.transpose(-1, -2))
        jitter = self.eps * torch.eye(self.n_out, device=cov.device, dtype=cov.dtype)
        # eigh on a batch of small near-degenerate matrices is fragile in
        # float32; double precision here costs little and stops the solver from
        # failing outright on repeated eigenvalues.
        values, vectors = torch.linalg.eigh((projected + jitter).double())
        values = torch.log(values.clamp_min(self.eps))
        logm = (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
        flat = logm[..., self.triu[0], self.triu[1]].to(cov.dtype)
        return flat * self.triu_scale


class WaveformBranch(nn.Module):
    """Depthwise-separable temporal encoder for one character epoch."""

    def __init__(self, n_channels: int, n_spatial: int = 48, embed_dim: int = 96,
                 dropout: float = 0.25) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, n_spatial, 1, bias=False)
        self.norm0 = nn.BatchNorm1d(n_spatial)
        # ~117 ms and ~55 ms kernels: an N400 complex and its finer structure.
        self.temporal = nn.Conv1d(n_spatial, n_spatial, 15, padding=7,
                                  groups=n_spatial, bias=False)
        self.mix = nn.Conv1d(n_spatial, 64, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(64)
        self.temporal2 = nn.Conv1d(64, 64, 7, padding=3, groups=64, bias=False)
        self.mix2 = nn.Conv1d(64, 96, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(96)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.norm0(self.spatial(x)))
        h = F.elu(self.norm1(self.mix(self.temporal(h))))
        h = self.drop(F.avg_pool1d(h, 2))
        h = F.elu(self.norm2(self.mix2(self.temporal2(h))))
        return self.proj(torch.cat([h.mean(-1), h.std(-1)], dim=-1))


class GeometryBranch(nn.Module):
    """Learnable xDAWN super-trial, covariance, and one SPDNet stage."""

    def __init__(self, n_channels: int, n_times: int, n_filters: int = 12,
                 n_prototypes: int = 6, n_manifold: int = 12,
                 embed_dim: int = 96) -> None:
        super().__init__()
        self.xdawn = LearnableXdawn(n_channels, n_filters, n_prototypes, n_times)
        self.cov = SpdCovariance()
        self.manifold = BiMapLogEig(self.xdawn.width, n_manifold)
        width = n_manifold * (n_manifold + 1) // 2
        self.proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.manifold(self.cov(self.xdawn(x))))


class PrecisionPool(nn.Module):
    """Inverse-variance accumulation over the characters of a row.

    Softmax attention asks which character matters most. With a 0.15 microvolt
    response in 10 microvolts of noise there is no such character, and attention
    pooling duly lost to the mean. This asks a different question -- how noisy
    is each character's estimate -- and combines them by precision, which is the
    minimum-variance unbiased combination of independent noisy estimates. The
    weights are non-competitive: one slot becoming reliable does not suppress
    the others, unlike softmax.
    """

    def __init__(self, embed_dim: int, max_log_precision: float = 3.0) -> None:
        super().__init__()
        self.log_precision = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.log_precision.weight)
        nn.init.zeros_(self.log_precision.bias)
        self.cap = float(max_log_precision)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Accumulate in representation space, not decision space. Reading each
        # character out to a scalar first and averaging the scalars keeps only
        # one projection of a 96-dimensional embedding and discards the rest;
        # measured at initialisation that collapsed the row output to a standard
        # deviation of 0.05 against a unit-variance target, and the model never
        # recovered. Pooling the embeddings leaves the head something to work
        # with, and the precision weights do the same job either way.
        log_precision = self.log_precision(z).squeeze(-1).clamp(-self.cap, self.cap)
        weight = torch.exp(log_precision) * mask
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bs,bsd->bd", weight, z), weight


class EvokedGeometryNet(nn.Module):
    """Two coupled views of each character epoch, accumulated by precision."""

    def __init__(self, spatial, *, n_channels: int = 128, n_times: int = 71,
                 embed_dim: int = 96, dropout: float = 0.25,
                 n_filters: int = 12, n_prototypes: int = 6,
                 n_manifold: int = 12, couple: bool = True,
                 use_geometry: bool = True, pool_mode: str = "precision") -> None:
        super().__init__()
        self.spatial = spatial
        self.use_geometry = use_geometry
        self.pool_mode = pool_mode
        self.waveform = WaveformBranch(n_channels, embed_dim=embed_dim, dropout=dropout)
        self.geometry = (
            GeometryBranch(n_channels, n_times, n_filters, n_prototypes,
                           n_manifold, embed_dim)
            if use_geometry else None
        )
        self.couple = couple and use_geometry
        if self.couple:
            # FiLM: the geometry view rescales and shifts the waveform view.
            # Zero-initialised, so training starts at the waveform branch alone
            # and any coupling has to be earned.
            self.film = nn.Linear(embed_dim, 2 * embed_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        fuse_in = 2 * embed_dim if use_geometry else embed_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in), nn.Linear(fuse_in, embed_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.pool = PrecisionPool(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor, group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial(eeg, group).reshape(b * s, c, t)
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
        if self.pool_mode == "mean":
            weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled = torch.einsum("bs,bsd->bd", weight, z)
        else:
            pooled, weight = self.pool(z, mask)
        row = self.head(pooled).squeeze(-1)
        return {"row": row, "char": torch.zeros_like(mask),
                "row_from_chars": row, "weights": weight, "embedding": pooled}


class EEGNetBag(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018) in the same bag framework, as a baseline.

    Same spatial front-end, same mean pooling over characters, same head, so a
    difference against it is a difference in the encoder and nothing else.
    """

    def __init__(self, spatial, *, n_channels: int = 128, n_times: int = 71,
                 f1: int = 8, depth: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.spatial = spatial
        f2 = f1 * depth
        self.conv = nn.Conv2d(1, f1, (1, 33), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable = nn.Conv2d(f2, f2, (1, 9), padding=(0, 4), groups=f2, bias=False)
        self.pointwise = nn.Conv2d(f2, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(1))

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor, group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial(eeg, group).reshape(b * s, 1, c, t)
        h = self.bn1(self.conv(x))
        h = self.drop1(self.pool1(F.elu(self.bn2(self.depthwise(h)))))
        h = self.drop2(self.pool2(F.elu(self.bn3(self.pointwise(self.separable(h))))))
        value = self.head(h).reshape(b, s)
        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        row = (value * weight).sum(dim=1)
        return {"row": row, "char": value, "row_from_chars": row,
                "weights": weight, "embedding": value}
