"""Abstract base for calibration strategies (Strategy pattern).

Every calibration strategy must implement:
  - fit(pairs_train)   → CalibrationModel
  - apply(line2d)      → corrected Line2D
  - evaluate(pairs_test) → dict of metrics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seis2cube.models.line2d import Line2D


@dataclass
class CalibrationPair:
    """Matched (2D, 3D) amplitude data at common (X,Y,t) locations.

    Attributes
    ----------
    coords : (N, 2) — world (X, Y).
    amp_2d : (N, S) — 2D amplitudes (traces or windows).
    amp_3d : (N, S) — corresponding 3D amplitudes.
    dt_ms  : sample interval.
    """
    coords: np.ndarray
    amp_2d: np.ndarray
    amp_3d: np.ndarray
    dt_ms: float


@dataclass
class CalibrationModel:
    """Serialisable container for fitted calibration parameters."""
    method: str
    params: dict[str, Any] = field(default_factory=dict)


class CalibrationStrategy(ABC):
    """Interface for all calibration methods (2D → 3D amplitude matching)."""

    @abstractmethod
    def fit(self, pairs: CalibrationPair) -> CalibrationModel:
        """Learn calibration parameters from matched 2D/3D pairs."""
        ...

    @abstractmethod
    def apply(self, line: Line2D, model: CalibrationModel) -> Line2D:
        """Apply calibration to a full 2D line, returning a corrected copy."""
        ...

    def evaluate(self, pairs: CalibrationPair, model: CalibrationModel) -> dict[str, float]:
        """Evaluate calibration on test pairs.  Default implementation."""
        corrected = self._apply_array(pairs.amp_2d, model)
        ref = pairs.amp_3d

        # Per-trace correlation
        corrs = []
        for i in range(len(corrected)):
            c = np.corrcoef(corrected[i], ref[i])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
        mean_corr = float(np.mean(corrs)) if corrs else 0.0

        rmse = float(np.sqrt(np.mean((corrected - ref) ** 2)))
        mae = float(np.mean(np.abs(corrected - ref)))

        # Spectral difference
        from seis2cube.utils.spectral import amplitude_spectrum
        _, spec_corr = amplitude_spectrum(corrected, pairs.dt_ms)
        _, spec_ref = amplitude_spectrum(ref, pairs.dt_ms)
        mean_spec_corr = spec_corr.mean(axis=0)
        mean_spec_ref = spec_ref.mean(axis=0)
        denom = np.linalg.norm(mean_spec_ref)
        spec_diff = float(np.linalg.norm(mean_spec_corr - mean_spec_ref) / max(denom, 1e-30))

        return {
            "pearson_corr": mean_corr,
            "rmse": rmse,
            "mae": mae,
            "spectral_l2_rel": spec_diff,
        }

    @abstractmethod
    def _apply_array(self, amp_2d: np.ndarray, model: CalibrationModel) -> np.ndarray:
        """Apply calibration to a raw 2D amplitude array.  Internal helper."""
        ...
