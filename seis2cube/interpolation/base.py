"""Abstract base for interpolation strategies (Strategy pattern).

Each strategy implements:
  - fit(calibration_cube3d, sampling_simulator)  → internal state / hyperparams
  - reconstruct(sparse_volume) → full volume (np.ndarray)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from seis2cube.models.volume import SparseVolume


@dataclass
class InterpolationResult:
    """Container for interpolation output + diagnostics."""
    volume: np.ndarray          # (n_il, n_xl, n_samp) filled volume
    cost_history: list[float] | None = None  # convergence cost per iteration (POCS etc.)
    metrics: dict[str, float] | None = None


class InterpolationStrategy(ABC):
    """Interface for all spatial interpolation / reconstruction methods."""

    @abstractmethod
    def fit(
        self,
        full_volume: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float]:
        """Tune hyper-parameters by simulating sparse reconstruction inside 3D.

        Parameters
        ----------
        full_volume : (n_il, n_xl, n_samp) — the known 3D cube (ground truth).
        mask : (n_il, n_xl) bool — simulated observation mask (True = observed).

        Returns
        -------
        metrics : reconstruction quality on the masked-out positions.
        """
        ...

    @abstractmethod
    def reconstruct(self, sparse: SparseVolume) -> InterpolationResult:
        """Reconstruct a full 3D volume from sparse observations.

        Parameters
        ----------
        sparse : SparseVolume with NaN where data is missing.

        Returns
        -------
        InterpolationResult with the filled volume.
        """
        ...
