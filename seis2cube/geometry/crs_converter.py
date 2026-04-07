"""Coordinate Reference System conversion using pyproj."""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

from seis2cube.config import CRSConfig


class CRSConverter:
    """Converts coordinates between source and target CRS.

    If source_crs is None the converter acts as an identity (no-op).
    """

    def __init__(self, config: CRSConfig) -> None:
        self._cfg = config
        self._transformer: Transformer | None = None
        self._inverse: Transformer | None = None

        if config.source_crs is not None and config.source_crs != config.target_crs:
            src = CRS(config.source_crs)
            tgt = CRS(config.target_crs)
            self._transformer = Transformer.from_crs(src, tgt, always_xy=True)
            self._inverse = Transformer.from_crs(tgt, src, always_xy=True)

    @property
    def is_identity(self) -> bool:
        return self._transformer is None

    def forward(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Source CRS → Target CRS.  x, y are 1-D arrays."""
        if self._transformer is None:
            return x.copy(), y.copy()
        xo, yo = self._transformer.transform(x, y)
        return np.asarray(xo, dtype=np.float64), np.asarray(yo, dtype=np.float64)

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Target CRS → Source CRS."""
        if self._inverse is None:
            return x.copy(), y.copy()
        xo, yo = self._inverse.transform(x, y)
        return np.asarray(xo, dtype=np.float64), np.asarray(yo, dtype=np.float64)

    def forward_point(self, x: float, y: float) -> tuple[float, float]:
        if self._transformer is None:
            return x, y
        return self._transformer.transform(x, y)

    def inverse_point(self, x: float, y: float) -> tuple[float, float]:
        if self._inverse is None:
            return x, y
        return self._inverse.transform(x, y)
