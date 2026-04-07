"""GeometryModel3D: mapping between (iline, xline) ↔ (X, Y) and spatial lookup.

Two strategies:
  - AffineGridMapper  — fast, assumes perfectly regular inline/xline grid.
  - KDTreeMapper      — fallback for irregular/warped grids; uses scipy.spatial.cKDTree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree


class GeometryModel3D(ABC):
    """Abstract base for 3D geometry lookup."""

    @abstractmethod
    def xy_to_ilxl(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map world (X, Y) → fractional (inline, xline)."""
        ...

    @abstractmethod
    def ilxl_to_xy(self, il: np.ndarray, xl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map (inline, xline) → world (X, Y)."""
        ...

    @abstractmethod
    def nearest_traces(
        self, x: np.ndarray, y: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, trace_indices) for k nearest 3D traces to each (x,y)."""
        ...


class AffineGridMapper(GeometryModel3D):
    """Assumes inline/xline form a regular affine grid in XY space.

    Parameters
    ----------
    coords : (N, 2) array of (X, Y) for all 3D traces, ordered inline-major.
    inlines, xlines : 1-D sorted unique arrays of inline/crossline labels.
    """

    def __init__(
        self,
        coords: np.ndarray,
        inlines: np.ndarray,
        xlines: np.ndarray,
    ) -> None:
        self._inlines = np.asarray(inlines, dtype=np.float64)
        self._xlines = np.asarray(xlines, dtype=np.float64)
        self._coords = np.asarray(coords, dtype=np.float64)  # (N, 2)

        n_il, n_xl = len(inlines), len(xlines)

        # Estimate affine from first three corners:
        #   corner (0,0), (0, n_xl-1), (n_il-1, 0)
        p00 = coords[0]
        p0n = coords[n_xl - 1]
        pn0 = coords[(n_il - 1) * n_xl]

        # A @ [il_frac, xl_frac].T + b = [x, y].T
        # il_frac = (il - il0) / dil, xl_frac = (xl - xl0) / dxl
        self._il0 = float(inlines[0])
        self._xl0 = float(xlines[0])
        self._dil = float(inlines[1] - inlines[0]) if n_il > 1 else 1.0
        self._dxl = float(xlines[1] - xlines[0]) if n_xl > 1 else 1.0

        # Affine columns
        col_il = (pn0 - p00) / max(n_il - 1, 1)  # dx per il-index step
        col_xl = (p0n - p00) / max(n_xl - 1, 1)  # dx per xl-index step
        self._A = np.column_stack([col_il, col_xl])  # (2, 2)
        self._b = p00.copy()
        self._A_inv = np.linalg.inv(self._A)

        # KDTree for nearest-trace queries
        self._tree = cKDTree(self._coords)

    def xy_to_ilxl(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack([np.asarray(x, dtype=np.float64),
                               np.asarray(y, dtype=np.float64)])
        frac = (pts - self._b) @ self._A_inv.T  # (N, 2): col0=il_idx, col1=xl_idx
        il = frac[:, 0] * self._dil + self._il0
        xl = frac[:, 1] * self._dxl + self._xl0
        return il, xl

    def ilxl_to_xy(self, il: np.ndarray, xl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        il_idx = (np.asarray(il, dtype=np.float64) - self._il0) / self._dil
        xl_idx = (np.asarray(xl, dtype=np.float64) - self._xl0) / self._dxl
        frac = np.column_stack([il_idx, xl_idx])
        pts = frac @ self._A.T + self._b
        return pts[:, 0], pts[:, 1]

    def nearest_traces(
        self, x: np.ndarray, y: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack([np.asarray(x), np.asarray(y)])
        dists, idxs = self._tree.query(pts, k=k)
        return np.atleast_2d(dists), np.atleast_2d(idxs)


class KDTreeMapper(GeometryModel3D):
    """Fallback geometry mapper using cKDTree — works for any coordinate layout.

    Parameters
    ----------
    coords : (N, 2) float array of (X, Y).
    il_labels, xl_labels : (N,) int arrays of per-trace inline/xline values.
    """

    def __init__(
        self,
        coords: np.ndarray,
        il_labels: np.ndarray,
        xl_labels: np.ndarray,
    ) -> None:
        self._coords = np.asarray(coords, dtype=np.float64)
        self._il = np.asarray(il_labels, dtype=np.float64)
        self._xl = np.asarray(xl_labels, dtype=np.float64)
        self._tree = cKDTree(self._coords)

    def xy_to_ilxl(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approximate inverse via nearest-neighbour lookup."""
        pts = np.column_stack([np.asarray(x), np.asarray(y)])
        _, idxs = self._tree.query(pts, k=1)
        return self._il[idxs], self._xl[idxs]

    def ilxl_to_xy(self, il: np.ndarray, xl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward lookup: find traces matching (il, xl) exactly or nearest."""
        # Build a second tree in (il, xl) space
        ilxl_tree = cKDTree(np.column_stack([self._il, self._xl]))
        query = np.column_stack([np.asarray(il, dtype=np.float64),
                                 np.asarray(xl, dtype=np.float64)])
        _, idxs = ilxl_tree.query(query, k=1)
        return self._coords[idxs, 0], self._coords[idxs, 1]

    def nearest_traces(
        self, x: np.ndarray, y: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack([np.asarray(x), np.asarray(y)])
        dists, idxs = self._tree.query(pts, k=k)
        return np.atleast_2d(dists), np.atleast_2d(idxs)
