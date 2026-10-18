"""Unrolled deconvolution conditioned on the stimulus schedule.

The gap this fills
------------------
Published EEG architectures treat overlapping evoked responses implicitly. The
multi-scale convolution plus cross-scale attention family -- MSVTNet (IEEE JBHI
2024, 28(12)), EEG Conformer (IEEE TNSRE 2023, 31:710-719) -- and the pretrained
backbones (EEGPT, NeurIPS 2024; CBraMod, ICLR 2025) all feed overlapping windows
to a generic encoder and let attention discover whatever structure it can. RSVP
decoders exploit the phase-locked character of an ERP but still do not separate
one stimulus's response from its neighbours'.

Classical electrophysiology does separate them, by deconvolution: rERP and
temporal response functions regress the continuous signal on a lagged impulse
design. On this corpus that works -- the surprisal TRF recovers a clean N400 at
367 ms with a sustained component to 790 ms -- but it is linear, offline, and
cannot be trained jointly with a decoder.

This layer puts the deconvolution *inside* the network. Characters arrive every
420 ms by program schedule while the response runs to 790 ms, so each window
holds its own character's response superimposed on the next two. The schedule is
deterministic and known, so the contamination is a known set of time shifts
rather than something to be inferred: the layer subtracts each neighbour's
current latent response, shifted by the schedule, and re-encodes. Unrolling that
fixed point for a few steps gives a nonlinear generalisation of what a TRF does
in one linear solve.

Why this cannot leak duration
-----------------------------
The shift uses the *paradigm constant* 0.42 s, a property of the experiment, not
the per-row ``duration / n_slots``. Using the per-row value would import
presentation duration, which the leakage contract forbids and which a row-level
sequence model was already shown to exploit (single-reading rho inflated to 0.32
while item rho fell).

Falsifiable predictions
-----------------------
1. The zero-initialised gates must learn non-zero magnitude, or the layer is
   inert and the architecture reduces to its backbone.
2. The gain should grow with the number of neighbours inside the window, so a
   1200 ms window should benefit more than a 450 ms one.
3. Removing the layer should return the score to the backbone's.
"""
from __future__ import annotations

import torch
from torch import nn

from .modern import DropPath, RMSNorm

SOA_SECONDS = 0.42          # measured paradigm constant, not a per-row quantity
SAMPLE_RATE = 128.0


def schedule_shift_tokens(stride: int, soa_seconds: float = SOA_SECONDS) -> int:
    """Neighbour offset in encoder tokens implied by the stimulus schedule."""
    return max(1, int(round(soa_seconds * SAMPLE_RATE / stride)))


class ScheduleDeconv(nn.Module):
    """One unrolled deconvolution step over the character axis.

    ``h`` is time-resolved per slot: ``[B, S, T, D]``. For slot ``k`` the layer
    gathers the latent responses of slots within ``radius``, shifts each by the
    schedule offset that separates them, sums them into a contamination estimate,
    and subtracts a gated projection of it. The gate is zero-initialised, so the
    block starts as the identity and any subtraction has to reduce the loss.
    """

    def __init__(self, d_model: int, shift: int, radius: int = 2,
                 dropout: float = 0.1, drop_path: float = 0.0) -> None:
        super().__init__()
        self.shift = int(shift)
        self.radius = int(radius)
        self.norm = RMSNorm(d_model)
        self.mix = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        # One learnable weight per neighbour offset: the response decays with
        # lag, so a neighbour two characters away should not be subtracted with
        # the same strength as the adjacent one.
        self.lag_weight = nn.Parameter(torch.ones(2 * self.radius))

    @staticmethod
    def _shift_time(x: torch.Tensor, amount: int) -> torch.Tensor:
        """Shift along the time axis, zero-filling what moves in from outside."""
        if amount == 0:
            return x
        out = torch.zeros_like(x)
        if amount > 0:
            out[..., amount:, :] = x[..., :-amount, :]
        else:
            out[..., :amount, :] = x[..., -amount:, :]
        return out

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h [B, S, T, D]; mask [B, S]
        b, s, t, d = h.shape
        valid = mask.view(b, s, 1, 1)
        normed = self.mix(self.norm(h)) * valid

        contamination = torch.zeros_like(h)
        index = 0
        for offset in range(-self.radius, self.radius + 1):
            if offset == 0:
                continue
            weight = self.lag_weight[index]
            index += 1
            # Slot k+offset's response reaches slot k's window displaced by
            # offset * schedule_shift tokens. Positive offset is a later
            # character, whose response starts later inside this window.
            if offset > 0:
                source = normed[:, offset:]
                target = slice(0, s - offset)
            else:
                source = normed[:, :s + offset]
                target = slice(-offset, s)
            shifted = self._shift_time(source, offset * self.shift)
            contamination[:, target] = contamination[:, target] + weight * shifted

        cleaned = h - self.drop_path(self.drop(self.gate(contamination)))
        return cleaned * valid

    @torch.no_grad()
    def gate_magnitude(self) -> float:
        """Diagnostic: how far the layer has moved from the identity."""
        return float(self.gate.weight.norm() + self.gate.bias.norm())


class ScheduleDeconvStack(nn.Module):
    """A few unrolled steps, each with its own gate."""

    def __init__(self, d_model: int, stride: int, steps: int = 2, radius: int = 2,
                 dropout: float = 0.1, drop_path: float = 0.05) -> None:
        super().__init__()
        shift = schedule_shift_tokens(stride)
        self.shift = shift
        self.steps = nn.ModuleList(
            ScheduleDeconv(d_model, shift, radius, dropout, drop_path)
            for _ in range(steps)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            h = step(h, mask)
        return h

    @torch.no_grad()
    def gate_magnitudes(self) -> list[float]:
        return [step.gate_magnitude() for step in self.steps]
