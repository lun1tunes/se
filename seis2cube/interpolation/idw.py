"""Baseline interpolation: Inverse Distance Weighting per time-slice.

Simple, fast, but typically produces "blurry" results between lines.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from seis2cube.interpolation.base import InterpolationResult, InterpolationStrategy
from seis2cube.models.volume import SparseVolume


class IDWTimeSliceInterpolator(InterpolationStrategy):
    """IDW interpolation applied independently to each time-slice.

    Parameters
    ----------
    power : IDW exponent (default 2).
    max_neighbours : max neighbours for weighted average.
    """

    def __init__(self, power: float = 2.0, max_neighbours: int = 12) -> None:
        self._power = power
        self._k = max_neighbours

    def fit(
        self,
        full_volume: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float]:
        """Simulate IDW on the 3D volume with given mask, return metrics."""
        sparse = self._mask_volume(full_volume, mask)
        recon = self._idw_volume(sparse, mask)
        return self._evaluate(full_volume, recon, mask)

    def reconstruct(self, sparse: SparseVolume) -> InterpolationResult:
        recon = self._idw_volume(sparse.data, sparse.mask)
        return InterpolationResult(volume=recon)

    # -- internals -----------------------------------------------------------

    def _idw_volume(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run IDW on each time-slice."""
        n_il, n_xl, n_samp = data.shape
        result = data.copy()

        # Build grid indices
        il_idx, xl_idx = np.meshgrid(np.arange(n_il), np.arange(n_xl), indexing="ij")
        obs_mask = mask.astype(bool)
        obs_coords = np.column_stack([il_idx[obs_mask], xl_idx[obs_mask]])
        miss_mask = ~obs_mask
        miss_coords = np.column_stack([il_idx[miss_mask], xl_idx[miss_mask]])

        if len(obs_coords) == 0 or len(miss_coords) == 0:
            return result

        tree = cKDTree(obs_coords)
        k_actual = min(self._k, len(obs_coords))
        dists, idxs = tree.query(miss_coords, k=k_actual)

        # Ensure 2D even for k=1 (cKDTree.query returns 1D when k=1)
        if k_actual == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        # Weights
        dists = np.clip(dists, 1e-10, None)
        weights = 1.0 / dists ** self._power
        weights /= weights.sum(axis=1, keepdims=True)

        # Observed trace indices (flat into obs arrays)
        obs_il = obs_coords[:, 0]
        obs_xl = obs_coords[:, 1]

        logger.info("IDW: interpolating {} missing positions from {} observed", len(miss_coords), len(obs_coords))

        for t in range(n_samp):
            obs_vals = data[obs_il, obs_xl, t]
            interp_vals = np.sum(weights * obs_vals[idxs], axis=1)
            result[miss_coords[:, 0], miss_coords[:, 1], t] = interp_vals

        return result

    @staticmethod
    def _mask_volume(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = vol.copy()
        out[~mask] = np.nan
        return out

    @staticmethod
    def _evaluate(truth: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        miss = ~mask
        t = truth[miss]
        r = recon[miss]
        valid = np.isfinite(r) & np.isfinite(t)
        t, r = t[valid], r[valid]
        rmse = float(np.sqrt(np.mean((t - r) ** 2)))
        corr = float(np.corrcoef(t, r)[0, 1]) if len(t) > 1 else 0.0
        mae = float(np.mean(np.abs(t - r)))
        return {"rmse": rmse, "mae": mae, "pearson_corr": corr}
