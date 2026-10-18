"""Hierarchical window and slot encoders for row-level surprisal regression.

The primary MM_mid configuration has separate EEG and eye temporal encoders,
a zero-initialized linear eye residual, cross-slot integration and masked mean
row readout. Optional experimental modules are not enabled by default.

Archived slot centers were estimated from row duration, and masks reveal slot
count. Fixed-width epochs do not establish timing independence or verified
character alignment. The README documents the archived normalization policy,
retrospective evaluation and separate physical/fixed-grid controls.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .modern import ModernStack
from .seqnet import MinimalSSM


class SinusoidalTime(nn.Module):
    """Fixed positional code for the within-character time axis."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


def _block(kind: str, d_model: int, n_heads: int, dropout: float, depth: int,
           rotary: bool = True) -> nn.Module:
    if kind == "transformer":
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=depth)
    if kind == "modern":
        # Rotary position, LayerScale, DropPath, RMSNorm and SwiGLU at matched
        # parameter count. Absolute position codes are the wrong prior here: the
        # structure is relative latency from character onset.
        return ModernStack(d_model, depth, n_heads, dropout, rotary=rotary)
    if kind == "ssm":
        return nn.ModuleList(
            nn.ModuleList([nn.LayerNorm(d_model), MinimalSSM(d_model)])
            for _ in range(depth)
        )
    raise ValueError(f"Unknown sequence block: {kind}")


def _apply(block: nn.Module, h: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    if isinstance(block, ModernStack):
        # Rotary attention carries position itself, so no additive code is added
        # before this block.
        return block(h, pad_mask)
    if isinstance(block, nn.TransformerEncoder):
        return block(h, src_key_padding_mask=pad_mask)
    for norm, ssm in block:
        h = h + ssm(norm(h))
        if pad_mask is not None:
            h = h * (~pad_mask).unsqueeze(-1).to(h.dtype)
    return h


class CharTimeEncoder(nn.Module):
    """Convolutional stem plus a sequence block over one epoch's time axis."""

    def __init__(self, n_channels: int = 128, n_spatial: int = 64, d_model: int = 96,
                 stride: int = 4, depth: int = 2, n_heads: int = 4,
                 block: str = "transformer", dropout: float = 0.1,
                 rotary: bool = True) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, n_spatial, 1, bias=False)
        self.norm0 = nn.BatchNorm1d(n_spatial)
        # ~117 ms depthwise kernel: one N400 complex before any downsampling, so
        # the sequence block receives components rather than raw samples.
        self.temporal = nn.Conv1d(n_spatial, n_spatial, 15, padding=7,
                                  groups=n_spatial, bias=False)
        self.mix = nn.Conv1d(n_spatial, d_model, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.stride = stride
        self.drop = nn.Dropout(dropout)
        self.position = SinusoidalTime(d_model)
        self.block = _block(block, d_model, n_heads, dropout, depth, rotary)
        self.out = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))

    def forward(self, x: torch.Tensor, *, return_local: bool = False):
        h = F.elu(self.norm0(self.spatial(x)))
        h = F.elu(self.norm1(self.mix(self.temporal(h))))
        h = self.drop(F.avg_pool1d(h, self.stride)).transpose(1, 2)
        local_h = h
        if not (isinstance(self.block, ModernStack) and self.block.rotary):
            h = self.position(h)
        h = _apply(self.block, h, None)
        pooled = self.out(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1))
        return (pooled, local_h) if return_local else pooled


class OverlapDemixer(nn.Module):
    """Resolve fixed-latency overlap between adjacent character EEG windows.

    The source cache spans -100..+1200 ms and highlights arrive about every
    420 ms. Local convolutional features from the current middle/late interval
    are contrasted with the next character's early/middle interval at the same
    physical time. The next character is read from the *same row* only. No SOA,
    duration, character identity, or text feature enters this module.

    Both residual projections are zero initialized, making the initial model
    exactly the existing hierarchical encoder. This isolates the new mechanism
    during a matched comparison.
    """

    def __init__(self, d_model: int, *, sample_rate: float = 128.0,
                 stride: int = 4, pre_ms: float = 100.0,
                 lag_ms: float = 420.0) -> None:
        super().__init__()
        self.start = round(pre_ms * sample_rate / (1000 * stride))
        self.first = round((pre_ms + lag_ms) * sample_rate / (1000 * stride))
        self.second = round((pre_ms + 2 * lag_ms) * sample_rate / (1000 * stride))
        self.middle_projection = nn.Linear(d_model, d_model, bias=False)
        self.late_projection = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.middle_projection.weight)
        nn.init.zeros_(self.late_projection.weight)

    def forward(self, embedding: torch.Tensor, local: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # embedding [B,S,D]; local [B,S,T,D].
        if local.shape[2] <= self.second:
            raise ValueError("EEG window too short for overlap demixing")
        early = local[:, :, self.start:self.first].mean(dim=2)
        middle = local[:, :, self.first:self.second].mean(dim=2)
        late = local[:, :, self.second:].mean(dim=2)
        next_early = torch.zeros_like(early)
        next_middle = torch.zeros_like(middle)
        next2_early = torch.zeros_like(early)
        next_early[:, :-1] = early[:, 1:]
        next_middle[:, :-1] = middle[:, 1:]
        next2_early[:, :-2] = early[:, 2:]
        valid_next = torch.zeros_like(mask)
        valid_next[:, :-1] = mask[:, 1:]
        valid_next2 = torch.zeros_like(mask)
        valid_next2[:, :-2] = mask[:, 2:]
        middle_contrast = (middle - next_early) * valid_next[..., None]
        late_contrast = (late - next_middle - next2_early) * valid_next2[..., None]
        correction = (self.middle_projection(middle_contrast) +
                      self.late_projection(late_contrast))
        return embedding + correction * mask[..., None]


class CenteredEvidencePool(nn.Module):
    """Add count-normalized within-row EEG heterogeneity to the mean readout.

    A row-level target may depend on distributed neural responses that the
    masked mean erases. The module pools centered second and third moments of
    character embeddings, with finite-sample correction for row length. These
    centered quantities discard a row-constant shift before the added branch;
    it receives no book, count, timing, text, or identity feature. The output
    projection starts at zero so the existing row head remains the exact
    initial model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.project = nn.Linear(2 * d_model, d_model, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, pooled: torch.Tensor, characters: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (characters * mask[..., None]).sum(dim=1) / count
        centered = (characters - mean[:, None, :]) * mask[..., None]
        variance = centered.square().sum(dim=1) / (count - 1).clamp_min(1.0)
        third = centered.pow(3).sum(dim=1) / count
        spread = (variance + 1e-5).sqrt()
        asymmetry = torch.tanh(third / (variance + 1e-3).pow(1.5))
        return pooled + self.project(torch.cat((spread, asymmetry), dim=-1))


class FrozenChineseLanguagePrior(nn.Module):
    """Frozen causal GPT-2 blocks over EEG-derived character embeddings.

    No text, token id, language-model score, book, or reader is an input. A
    trainable input adapter maps EEG embeddings to the GPT-2 hidden width; a
    zero-initialized output adapter makes the candidate identical to the
    existing EEG model before training. The random arm has the same architecture
    and trainable parameter count, isolating the value of language pretraining.
    """

    MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"

    def __init__(self, d_model: int, *, mode: str, max_slots: int = 10,
                 n_layers: int = 2) -> None:
        super().__init__()
        if mode not in {"pretrained", "random"}:
            raise ValueError(f"Unknown language-prior mode: {mode}")
        from transformers import GPT2Config, GPT2Model
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        config = GPT2Config.from_pretrained(self.MODEL_ID, local_files_only=True)
        if n_layers < 1 or n_layers > config.n_layer:
            raise ValueError("Invalid number of GPT-2 layers")
        if max_slots > config.n_positions:
            raise ValueError("Too many character slots for GPT-2 positions")
        if mode == "pretrained":
            source = GPT2Model.from_pretrained(self.MODEL_ID, local_files_only=True)
            self.blocks = nn.ModuleList(list(source.h[:n_layers]))
            self.position = source.wpe
        else:
            self.blocks = nn.ModuleList(GPT2Block(config, layer_idx=i)
                                        for i in range(n_layers))
            self.position = nn.Embedding(config.n_positions, config.n_embd)
            nn.init.normal_(self.position.weight, std=config.initializer_range)
        self.blocks.requires_grad_(False)
        self.position.requires_grad_(False)
        self.blocks.eval()
        self.position.eval()
        # Loading a pretrained model and constructing random blocks consume
        # different RNG streams. Give their trainable adapters exactly the same
        # starting weights so the control isolates the frozen block weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7019)
            self.input_adapter = nn.Linear(d_model, config.n_embd, bias=False)
            self.input_norm = nn.LayerNorm(config.n_embd)
            self.output_adapter = nn.Linear(config.n_embd, d_model, bias=False)
            nn.init.zeros_(self.output_adapter.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.blocks.eval()
        self.position.eval()
        return self

    def forward(self, characters: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(characters.shape[1], device=characters.device)
        hidden = self.input_norm(self.input_adapter(characters)) + self.position(slots)[None]
        attention_mask = (1.0 - mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask,
                           use_cache=False)[0]
        return characters + self.output_adapter(hidden) * mask[..., None]


class HierSeqNet(nn.Module):
    """Within-character time block, then a cross-character block, then a head."""

    def __init__(self, spatial, *, n_channels: int = 128, d_model: int = 96,
                 n_spatial: int = 64, stride: int = 4, time_depth: int = 2,
                 char_depth: int = 2, n_heads: int = 4, max_slots: int = 10,
                 block: str = "transformer", dropout: float = 0.1,
                 use_char_block: bool = True,
                 rotary: bool = True) -> None:
        super().__init__()
        self.spatial_op = spatial
        self.time_encoder = CharTimeEncoder(
            n_channels, n_spatial, d_model, stride, time_depth, n_heads, block,
            dropout, rotary
        )
        self.use_char_block = use_char_block
        if use_char_block:
            # Slot position is learned, not sinusoidal: there are at most ten
            # slots and their spacing is a fixed stimulus schedule, so an
            # embedding is both cheaper and a better fit than a continuous code.
            self.slot_position = nn.Embedding(max_slots, d_model)
            nn.init.normal_(self.slot_position.weight, std=0.02)
            self.char_block = _block(block, d_model, n_heads, dropout, char_depth, rotary)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor,
                group: torch.Tensor) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial_op(eeg, group).reshape(b * s, c, t)
        z = self.time_encoder(x).reshape(b, s, -1)
        if self.use_char_block:
            slots = torch.arange(s, device=z.device).unsqueeze(0)
            z = z + self.slot_position(slots)
            pad = mask < 0.5
            # A row whose every slot is padding would make attention undefined;
            # that cannot happen here (bags hold at least two characters) but the
            # guard keeps the block safe if the cache ever changes.
            if bool(pad.all(dim=1).any()):
                pad = pad & ~pad.all(dim=1, keepdim=True)
            z = _apply(self.char_block, z, pad)
        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bs,bsd->bd", weight, z)
        row = self.head(pooled).squeeze(-1)
        return {"row": row, "char": torch.zeros_like(mask),
                "row_from_chars": row, "weights": weight, "embedding": pooled}


class EyeCharEncoder(nn.Module):
    """Sequence encoder for one character's eye-tracking window.

    Four channels (gaze x, gaze y, pupil, validity), sampled at 50 Hz.
    Archived Eye windows span -100..+800 ms; the primary EEG windows instead
    span -100..+1200 ms. Separate encoders pool these unequal time sequences
    before slot-level fusion. Estimated slot centers depend on row duration;
    fixed window width does not remove duration-related geometry. Other window
    widths are explicitly tested by run_window_alignment_cv.py.
    Validity travels as its own
    channel because gaze is missing for about 5% of samples (blinks, tracker
    loss) and zero is a legal gaze coordinate: the model has to be able to tell
    absence from centre.
    """

    def __init__(self, n_channels: int = 4, d_model: int = 96, depth: int = 1,
                 n_heads: int = 4, block: str = "transformer",
                 dropout: float = 0.1, stats: tuple | None = None,
                 rotary: bool = True) -> None:
        super().__init__()
        # The four channels live on incompatible scales -- pupil diameter is
        # about 3.8 mm, gaze is normalised screen coordinates near 0.5, validity
        # is 0/1 -- and 4.4% of epochs are all zero because the archive is
        # missing. Feeding that straight into BatchNorm made the SSM variant
        # diverge to NaN on every prediction. Standardisation uses train-split
        # statistics over VALID samples only, held as buffers.
        mean = torch.zeros(n_channels) if stats is None else torch.as_tensor(stats[0])
        std = torch.ones(n_channels) if stats is None else torch.as_tensor(stats[1])
        self.register_buffer("channel_mean", mean.float().view(1, -1, 1))
        self.register_buffer("channel_std", std.float().clamp_min(1e-3).view(1, -1, 1))
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 32, 5, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.Conv1d(32, d_model, 3, padding=1, bias=False),
            nn.BatchNorm1d(d_model), nn.ELU(),
        )
        self.position = SinusoidalTime(d_model)
        self.block = _block(block, d_model, n_heads, dropout, depth, rotary)
        self.out = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validity (last channel) stays raw: it is already 0/1 and its whole job
        # is to mark absence, which standardisation would blur.
        scaled = (x - self.channel_mean) / self.channel_std
        x = torch.cat([scaled[:, :-1], x[:, -1:]], dim=1)
        h = self.stem(x).transpose(1, 2)
        if not (isinstance(self.block, ModernStack) and self.block.rotary):
            h = self.position(h)
        h = _apply(self.block, h, None)
        return self.out(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1))


class MultiModalHierSeqNet(nn.Module):
    """HierSeqNet with an eye branch fused per character, as one model.

    The eye stream enters at the character level and shares the cross-character
    block, so this stays a single jointly trained network rather than two models
    averaged afterwards. Fusion is a gated residual: the eye embedding is added
    to the EEG embedding through a zero-initialised gate, so training starts
    exactly at the EEG-only model and any eye contribution has to be earned. That
    matters here because the frozen experiment found row-level EEG+eye fusion did
    not beat EEG alone, and a fusion that cannot fall back is how that happens.
    """

    def __init__(self, spatial, *, n_channels: int = 128, d_model: int = 96,
                 n_spatial: int = 64, stride: int = 4, time_depth: int = 2,
                 char_depth: int = 2, n_heads: int = 4, max_slots: int = 10,
                 block: str = "transformer", dropout: float = 0.1,
                 eye_channels: int = 4, eye_depth: int = 1,
                 use_eye: bool = True, use_char_block: bool = True,
                 eye_stats: tuple | None = None, late_fusion: bool = False,
                 rotary: bool = True, overlap_demix: bool = False,
                 centered_evidence: bool = False,
                 llm_prior: str = "none") -> None:
        super().__init__()
        self.spatial_op = spatial
        self.use_eye = use_eye
        self.use_char_block = use_char_block
        self.time_encoder = CharTimeEncoder(
            n_channels, n_spatial, d_model, stride, time_depth, n_heads, block,
            dropout, rotary
        )
        if use_eye:
            self.eye_encoder = EyeCharEncoder(eye_channels, d_model, eye_depth,
                                              n_heads, block, dropout, eye_stats,
                                              rotary)
            self.eye_gate = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.eye_gate.weight)
            nn.init.zeros_(self.eye_gate.bias)
        if use_char_block:
            self.slot_position = nn.Embedding(max_slots, d_model)
            nn.init.normal_(self.slot_position.weight, std=0.02)
            self.char_block = _block(block, d_model, n_heads, dropout, char_depth, rotary)
            if late_fusion and use_eye:
                # Separate cross-character dynamics per modality. Gaze carries
                # return sweeps and fixation rhythm; EEG carries load
                # accumulation. Sharing one block forces one temporal structure
                # on both, which is the thing this variant tests.
                self.eye_slot_position = nn.Embedding(max_slots, d_model)
                nn.init.normal_(self.eye_slot_position.weight, std=0.02)
                self.eye_char_block = _block(block, d_model, n_heads, dropout, char_depth, rotary)
        self.late_fusion = late_fusion and use_eye
        if self.late_fusion:
            self.row_gate = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.row_gate.weight)
            nn.init.zeros_(self.row_gate.bias)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )
        # Instantiate after every shared layer so an equal seed gives the
        # baseline and candidate identical shared initial weights.
        self.overlap_demix = OverlapDemixer(d_model) if overlap_demix else None
        self.centered_evidence = CenteredEvidencePool(d_model) if centered_evidence else None
        self.language_prior = (None if llm_prior == "none" else
                               FrozenChineseLanguagePrior(d_model, mode=llm_prior,
                                                          max_slots=max_slots))

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor, group: torch.Tensor,
                eye: torch.Tensor | None = None) -> dict:
        b, s, c, t = eeg.shape
        x = self.spatial_op(eeg, group).reshape(b * s, c, t)
        if self.overlap_demix is not None:
            z, local = self.time_encoder(x, return_local=True)
            z = self.overlap_demix(z.reshape(b, s, -1),
                                   local.reshape(b, s, local.shape[1], -1), mask)
            z = z.reshape(b * s, -1)
        else:
            z = self.time_encoder(x)
        e = None
        if self.use_eye and eye is not None:
            e = self.eye_encoder(eye.reshape(b * s, eye.shape[2], eye.shape[3]))
            if not self.late_fusion:
                z = z + self.eye_gate(e)
        z = z.reshape(b, s, -1)
        if self.language_prior is not None:
            z = self.language_prior(z, mask)
        evidence_source = z
        pad = mask < 0.5
        if bool(pad.all(dim=1).any()):
            pad = pad & ~pad.all(dim=1, keepdim=True)
        slots = torch.arange(s, device=z.device).unsqueeze(0)
        if self.use_char_block:
            z = _apply(self.char_block, z + self.slot_position(slots), pad)
        weight = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bs,bsd->bd", weight, z)
        if self.late_fusion and e is not None:
            eye_z = e.reshape(b, s, -1)
            if self.use_char_block:
                eye_z = _apply(self.eye_char_block,
                               eye_z + self.eye_slot_position(slots), pad)
            eye_pooled = torch.einsum("bs,bsd->bd", weight, eye_z)
            pooled = pooled + self.row_gate(eye_pooled)
        if self.centered_evidence is not None:
            pooled = self.centered_evidence(pooled, evidence_source, mask)
        row = self.head(pooled).squeeze(-1)
        return {"row": row, "char": torch.zeros_like(mask),
                "row_from_chars": row, "weights": weight, "embedding": pooled}
