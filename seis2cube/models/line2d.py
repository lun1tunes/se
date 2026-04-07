"""Data model for a 2D seismic profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Line2D:
    """Container for a single 2D post-stack SEG-Y profile.

    Attributes
    ----------
    name : human-readable line identifier.
    path : original SEG-Y file path.
    coords : (N_traces, 2) array of (X, Y) in target CRS.
    data : (N_traces, N_samples) amplitude array.
    dt_ms : sample interval in milliseconds.
    delrt_ms : delay recording time (first sample time) in ms.
    quality_flags : optional per-trace quality indicators.
    """

    name: str
    path: Path
    coords: np.ndarray            # (N, 2)
    data: np.ndarray              # (N, S)
    dt_ms: float
    delrt_ms: float = 0.0
    quality_flags: np.ndarray | None = None

    # -- derived properties --------------------------------------------------

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def time_axis_ms(self) -> np.ndarray:
        """Time axis in ms."""
        return self.delrt_ms + np.arange(self.n_samples) * self.dt_ms

    # -- utilities -----------------------------------------------------------

    def resample(self, target_dt_ms: float, target_n_samples: int) -> "Line2D":
        """Resample data to a new sample interval / trace length via linear interp."""
        from scipy.interpolate import interp1d

        src_t = self.time_axis_ms
        dst_t = self.delrt_ms + np.arange(target_n_samples) * target_dt_ms

        f = interp1d(src_t, self.data, axis=1, kind="linear",
                     bounds_error=False, fill_value=0.0)
        new_data = f(dst_t).astype(np.float32)

        return Line2D(
            name=self.name,
            path=self.path,
            coords=self.coords.copy(),
            data=new_data,
            dt_ms=target_dt_ms,
            delrt_ms=self.delrt_ms,
            quality_flags=self.quality_flags,
        )

    def subset(self, indices: np.ndarray) -> "Line2D":
        """Return a new Line2D containing only selected trace indices."""
        return Line2D(
            name=self.name,
            path=self.path,
            coords=self.coords[indices],
            data=self.data[indices],
            dt_ms=self.dt_ms,
            delrt_ms=self.delrt_ms,
            quality_flags=self.quality_flags[indices] if self.quality_flags is not None else None,
        )
