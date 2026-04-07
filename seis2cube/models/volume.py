"""Data models for 3D grids and sparse volumes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TargetGrid:
    """Definition of the output 3D regular grid.

    The grid is defined by inline/xline ranges and a time axis,
    plus an affine mapping to world coordinates.
    """

    inlines: np.ndarray       # sorted unique inline labels
    xlines: np.ndarray        # sorted unique crossline labels
    n_samples: int
    dt_ms: float
    delrt_ms: float = 0.0

    # Affine parameters for (il_idx, xl_idx) → (X, Y)
    origin_x: float = 0.0
    origin_y: float = 0.0
    il_step_x: float = 0.0
    il_step_y: float = 0.0
    xl_step_x: float = 0.0
    xl_step_y: float = 0.0

    @property
    def n_il(self) -> int:
        return len(self.inlines)

    @property
    def n_xl(self) -> int:
        return len(self.xlines)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_il, self.n_xl, self.n_samples)

    @property
    def time_axis_ms(self) -> np.ndarray:
        return self.delrt_ms + np.arange(self.n_samples) * self.dt_ms

    def il_index(self, il: int) -> int:
        """Convert inline label → index in the grid."""
        idx = np.searchsorted(self.inlines, il)
        if idx >= len(self.inlines) or self.inlines[idx] != il:
            raise KeyError(f"Inline {il} not in grid")
        return int(idx)

    def xl_index(self, xl: int) -> int:
        idx = np.searchsorted(self.xlines, xl)
        if idx >= len(self.xlines) or self.xlines[idx] != xl:
            raise KeyError(f"Xline {xl} not in grid")
        return int(idx)

    def xy_at(self, il_idx: int, xl_idx: int) -> tuple[float, float]:
        """World coordinates at grid position."""
        x = self.origin_x + il_idx * self.il_step_x + xl_idx * self.xl_step_x
        y = self.origin_y + il_idx * self.il_step_y + xl_idx * self.xl_step_y
        return x, y


@dataclass
class SparseVolume:
    """A 3D volume where only some (iline, xline) positions have data.

    Attributes
    ----------
    grid : the target grid definition.
    data : (n_il, n_xl, n_samples) float32 array (NaN where missing).
    mask : (n_il, n_xl) bool array — True where data is observed.
    """

    grid: TargetGrid
    data: np.ndarray    # (n_il, n_xl, n_samples)
    mask: np.ndarray    # (n_il, n_xl) bool

    @classmethod
    def empty(cls, grid: TargetGrid) -> "SparseVolume":
        data = np.full(grid.shape, np.nan, dtype=np.float32)
        mask = np.zeros((grid.n_il, grid.n_xl), dtype=bool)
        return cls(grid=grid, data=data, mask=mask)

    def insert_trace(self, il_idx: int, xl_idx: int, trace: np.ndarray) -> None:
        """Insert a single trace into the sparse volume."""
        self.data[il_idx, xl_idx, :len(trace)] = trace[:self.grid.n_samples]
        self.mask[il_idx, xl_idx] = True

    @property
    def fill_ratio(self) -> float:
        return float(self.mask.sum()) / max(self.mask.size, 1)
