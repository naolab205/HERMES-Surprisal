"""Euclidean Alignment for cross-participant EEG.

Each participant's sensor covariance differs because of cap placement, skull
conductivity and impedance, so an encoder trained across participants spends
capacity undoing that instead of decoding the task.  Euclidean Alignment
whitens every participant by their own mean epoch covariance, mapping all
participants onto a common reference where the mean covariance is the
identity (He and Wu, 2020).

The reference covariance is estimated on TRAINING epochs only and then applied
unchanged to validation and test epochs of the same participant, so no label or
test-split information enters the transform.
"""
from __future__ import annotations

import numpy as np


def mean_covariance(epochs: np.ndarray, *, shrinkage: float = 0.05) -> np.ndarray:
    """Mean spatial covariance over epochs, with a little ridge shrinkage.

    ``epochs`` is ``[n_epochs, n_channels, n_times]``.  Shrinkage toward a
    scaled identity keeps the inverse square root well conditioned; the EEG is
    average-referenced, which makes the raw covariance exactly rank deficient.
    """
    x = np.asarray(epochs, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected [epochs, channels, times], got {x.shape}")
    n_times = x.shape[2]
    cov = np.einsum("nct,ndt->cd", x, x) / (len(x) * n_times)
    trace = float(np.trace(cov)) / cov.shape[0]
    return (1.0 - shrinkage) * cov + shrinkage * trace * np.eye(cov.shape[0])


def inverse_sqrt(cov: np.ndarray, *, eps: float = 1e-10,
                 rank_tol: float | None = None) -> np.ndarray:
    """Symmetric inverse square root, optionally restricted to the true rank.

    This EEG is average referenced *and* has its bad channels interpolated, and
    interpolation writes each repaired channel as a linear combination of its
    neighbours. Every interpolated channel therefore removes one more rank from
    the spatial covariance, on top of the one the average reference removes:
    measured on one recording, five eigenvalues sit at ~1e-6 against a largest
    of ~2.3e3.

    Whitening such a matrix amplifies those empty directions by the inverse
    square root of a nearly-zero eigenvalue -- about 44,000x with no shrinkage,
    and still 23x at the 0.05 shrinkage used earlier. That is the likely reason
    Euclidean Alignment on its own hurt. With ``rank_tol`` set, directions below
    ``rank_tol * max_eigenvalue`` are projected out instead of whitened, which
    is the rank-aware form the data calls for.
    """
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    largest = max(float(values.max()), eps)
    if rank_tol is not None:
        keep = values > rank_tol * largest
        values, vectors = values[keep], vectors[:, keep]
        return (vectors * (values ** -0.5)) @ vectors.T
    values = np.maximum(values, largest * eps)
    return (vectors * (values ** -0.5)) @ vectors.T


class EuclideanAligner:
    """Per-group whitening transform fitted on training epochs only.

    A "group" is normally one ``(subject, book)`` recording: cap placement is
    re-done between sessions, so a per-session reference is the honest unit.
    """

    def __init__(self, shrinkage: float = 0.05) -> None:
        self.shrinkage = float(shrinkage)
        self.transforms: dict[str, np.ndarray] = {}

    def fit_group(self, group: str, train_epochs: np.ndarray) -> None:
        cov = mean_covariance(train_epochs, shrinkage=self.shrinkage)
        self.transforms[str(group)] = inverse_sqrt(cov).astype(np.float32)

    def transform(self, group: str, epochs: np.ndarray) -> np.ndarray:
        matrix = self.transforms.get(str(group))
        if matrix is None:
            raise KeyError(f"No alignment fitted for group {group!r}")
        return np.einsum("cd,ndt->nct", matrix, np.asarray(epochs, dtype=np.float32))

    @property
    def fitted_groups(self) -> tuple[str, ...]:
        return tuple(sorted(self.transforms))
