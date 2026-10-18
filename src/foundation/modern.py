"""Modern transformer components, chosen for regularisation rather than capacity.

Capacity has been measured harmful on this corpus four separate times: width x8
(trial rho 0.212 -> 0.190), depth 2 -> 3 (item rho 0.389 -> 0.358), softmax
attention pooling (-0.010), and the geometry branch (-0.013 with a convolutional
backbone, -0.034 with a sequence backbone). The pattern is an overfitting
signature at a 1:70 single-trial signal-to-noise ratio, so the modules worth
importing are the ones that change inductive bias or optimisation at equal
parameter count -- not the ones that add parameters.

What is here and the specific reason each earns its place:

* **Rotary position embedding.** The baseline uses absolute sinusoidal codes, but
  the structure of this signal is *relative* latency: a response at tau
  milliseconds after character onset. RoPE encodes relative offsets directly in
  the attention dot product, which fixes a conceptual mismatch rather than adding
  machinery. Parameter-free.
* **LayerScale.** A per-branch learnable scalar, initialised near zero, so a
  residual block starts as the identity and depth has to earn its contribution.
  This is the standard remedy for exactly the failure observed at depth 3.
  One scalar per channel per branch.
* **DropPath.** Stochastic depth, the standard regulariser for transformers on
  small data. Parameter-free.
* **RMSNorm and SwiGLU.** Better-conditioned substitutes for LayerNorm and a
  GELU feed-forward at matched parameter count; SwiGLU's hidden width is reduced
  to 2/3 to keep the count equal.

Deliberately not here: more layers, more heads, wider models, mixture-of-experts.
Every one of those adds capacity, which is the thing that has repeatedly failed.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale * self.weight


class DropPath(nn.Module):
    """Drop an entire residual branch for a whole sample, with probability p."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def rotary_tables(length: int, head_dim: int, device, dtype):
    half = head_dim // 2
    freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / max(half, 1))
    )
    angle = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1) * freq
    return torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels by a position-dependent angle. x is [B, H, T, D]."""
    first, second = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([first * cos - second * sin,
                           first * sin + second * cos], dim=-1)
    return rotated.flatten(-2)


class RotaryAttention(nn.Module):
    """Self-attention with rotary position, or without it for the ablation.

    ``rotary=False`` removes ONLY the position mechanism and keeps RMSNorm,
    SwiGLU, LayerScale and DropPath in place, so the comparison isolates the
    position prior. Swapping the whole block for a vanilla TransformerEncoder
    would change five things at once, which is how the first version of this
    ablation was mislabelled.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 rotary: bool = True) -> None:
        super().__init__()
        self.rotary = bool(rotary)
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        if self.rotary:
            cos, sin = rotary_tables(t, self.head_dim, x.device, x.dtype)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        attn_mask = None
        if pad_mask is not None:
            attn_mask = (~pad_mask).view(b, 1, 1, t)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expand: float = 4 / 3) -> None:
        super().__init__()
        # 2/3 of a standard 2x GELU feed-forward, so the three projections here
        # match the two projections there in parameter count.
        hidden = int(round(d_model * expand))
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ModernBlock(nn.Module):
    """Pre-norm residual block: RMSNorm, rotary attention, SwiGLU, LayerScale, DropPath."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 drop_path: float = 0.1, layer_scale: float = 1e-4,
                 rotary: bool = True) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RotaryAttention(d_model, n_heads, dropout, rotary)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.scale1 = nn.Parameter(layer_scale * torch.ones(d_model))
        self.scale2 = nn.Parameter(layer_scale * torch.ones(d_model))
        self.drop_path = DropPath(drop_path)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.scale1 * self.attn(self.norm1(x), pad_mask))
        x = x + self.drop_path(self.scale2 * self.drop(self.ffn(self.norm2(x))))
        return x


class ModernStack(nn.Module):
    def __init__(self, d_model: int, depth: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, drop_path: float = 0.1,
                 layer_scale: float = 1e-4, rotary: bool = True) -> None:
        super().__init__()
        self.rotary = bool(rotary)
        # Linearly increasing drop rate with depth, as in the original
        # stochastic-depth schedule.
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            ModernBlock(d_model, n_heads, dropout, rate, layer_scale, rotary)
            for rate in rates
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.norm(x)
