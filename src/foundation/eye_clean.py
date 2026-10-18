"""Exploratory, training-only eye-to-EEG regression for ocular correction.

The eye cache covers -100..800 ms and the EEG cache -100..1200 ms. Correction
is confined to the aligned overlap. This is a nuisance-regression experiment,
not a reconstruction of the dataset authors' ICA pipeline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class EyeRegressionCleaner:
    def __init__(self, n_groups: int, device: torch.device, eeg_length: int,
                 eye_length: int = 45, ridge: float = 1000.0) -> None:
        self.device = device
        self.overlap = min(eeg_length, round(eye_length * 128 / 50))
        self.ridge = ridge
        self.coeff = torch.zeros(n_groups, 4, 128, device=device)
        self.mean = torch.zeros(n_groups, 3, device=device)
        self.std = torch.ones(n_groups, 3, device=device)
        self.fit_counts = torch.zeros(n_groups, dtype=torch.long, device=device)

    def features(self, eye: torch.Tensor, group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # eye: [N,4,45]; values are normalized gaze x/y, pupil, validity.
        physical = F.interpolate(eye[:, :3].float(), size=self.overlap,
                                 mode="linear", align_corners=False).transpose(1, 2)
        valid = F.interpolate(eye[:, 3:4].float(), size=self.overlap,
                              mode="nearest").squeeze(1) > .5
        z = (physical - self.mean[group, None, :]) / self.std[group, None, :]
        z = torch.cat((torch.ones_like(z[..., :1]), z), dim=-1)
        return z, valid

    @torch.no_grad()
    def fit_group(self, store, group_id: int, epoch_indices: torch.Tensor,
                  max_epochs: int = 1000) -> None:
        if epoch_indices.numel() > max_epochs:
            # Stable sampling independent of the model's training RNG.
            generator = torch.Generator(device=self.device).manual_seed(77001 + group_id)
            selected = torch.randperm(epoch_indices.numel(), device=self.device,
                                      generator=generator)[:max_epochs]
            epoch_indices = epoch_indices[selected]
        eye = store.eye.index_select(0, epoch_indices).float()
        eeg = store.data.index_select(0, epoch_indices).float()
        physical = F.interpolate(eye[:, :3], size=self.overlap, mode="linear",
                                 align_corners=False).transpose(1, 2)
        valid = F.interpolate(eye[:, 3:4], size=self.overlap,
                              mode="nearest").squeeze(1) > .5
        if valid.sum() < 1000:
            return
        x = physical[valid]
        mean = x.mean(0)
        std = x.std(0).clamp_min(1e-4)
        self.mean[group_id], self.std[group_id] = mean, std
        z = torch.cat((torch.ones_like(x[:, :1]), (x - mean) / std), dim=1)
        y = eeg[:, :, :self.overlap].transpose(1, 2)[valid]
        gram = z.T @ z
        penalty = torch.eye(4, device=self.device) * self.ridge
        penalty[0, 0] = 0.0
        coef = torch.linalg.solve(gram + penalty, z.T @ y)
        # Keep the intercept: subtract only eye-dependent variation.
        coef[0] = 0.0
        self.coeff[group_id] = coef
        self.fit_counts[group_id] = valid.sum()

    @torch.no_grad()
    def apply(self, eeg: torch.Tensor, eye: torch.Tensor,
              group: torch.Tensor) -> torch.Tensor:
        # eeg [B,S,128,T], eye [B,S,4,45], group [B].
        b, slots = eeg.shape[:2]
        flat_group = group[:, None].expand(b, slots).reshape(-1)
        z, valid = self.features(eye.reshape(-1, 4, eye.shape[-1]), flat_group)
        nuisance = torch.bmm(z, self.coeff[flat_group]).transpose(1, 2)
        nuisance = nuisance * valid[:, None, :]
        result = eeg.clone()
        result[..., :self.overlap] -= nuisance.reshape(b, slots, 128, self.overlap)
        return result
