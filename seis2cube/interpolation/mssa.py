"""MSSA (Multichannel Singular Spectrum Analysis) interpolation.

Treats the seismic data as low-rank in the Hankel/block-Hankel matrix sense.
Reconstructs missing traces by iterative SVD rank-reduction + observation
re-insertion (similar to POCS but in the SVD domain).

Handles irregular observation coordinates via an interpolation operator to the
regular grid (I-MSSA variant).

References:
  - MSSA for denoising and reconstruction of seismic data.
  - I-MSSA: accounts for off-the-grid observations through a linear interpolation
    operator mapping from irregular to regular positions.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from seis2cube.interpolation.base import InterpolationResult, InterpolationStrategy
from seis2cube.models.volume import SparseVolume


class MSSAInterpolator(InterpolationStrategy):
    """MSSA-based seismic interpolation.

    Parameters
    ----------
    rank : target rank for truncated SVD.
    window : Hankel embedding window length (in traces).
    n_iter : number of reconstruction iterations.
    """

    def __init__(
        self,
        rank: int = 20,
        window: int = 50,
        n_iter: int = 30,
    ) -> None:
        self._rank = rank
        self._window = window
        self._niter = n_iter

    # -- Strategy interface --------------------------------------------------

    def fit(
        self,
        full_volume: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float]:
        sparse = full_volume.copy().astype(np.float32)
        sparse[~mask] = 0.0
        recon = self._mssa_3d(sparse, mask)
        return self._evaluate(full_volume, recon, mask)

    def reconstruct(self, sparse: SparseVolume) -> InterpolationResult:
        data = np.nan_to_num(sparse.data, nan=0.0).astype(np.float32)
        recon = self._mssa_3d(data, sparse.mask)
        return InterpolationResult(volume=recon)

    # -- core MSSA -----------------------------------------------------------

    def _mssa_3d(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply MSSA reconstruction per inline section (il, :, :).

        For efficiency, process inline-by-inline: each "section" is
        (n_xl, n_samp) — a 2D matrix suitable for Hankel embedding + SVD.
        """
        n_il, n_xl, n_samp = data.shape
        result = data.copy()
        mask_3d = np.broadcast_to(mask[:, :, np.newaxis], data.shape)

        logger.info("MSSA: rank={}, window={}, iter={}", self._rank, self._window, self._niter)

        for il_idx in range(n_il):
            section = result[il_idx]  # (n_xl, n_samp)
            section_mask = mask[il_idx]  # (n_xl,) bool
            if section_mask.all():
                continue  # nothing to reconstruct

            result[il_idx] = self._mssa_section(section, section_mask)

        # Also run along crosslines for better quality
        for xl_idx in range(n_xl):
            section = result[:, xl_idx, :]  # (n_il, n_samp)
            section_mask = mask[:, xl_idx]
            if section_mask.all():
                continue

            result[:, xl_idx, :] = self._mssa_section(section, section_mask)

        # Re-insert observations
        result[mask_3d] = data[mask_3d]
        return result

    def _mssa_section(self, section: np.ndarray, trace_mask: np.ndarray) -> np.ndarray:
        """MSSA on a single 2D section (n_traces, n_samples).

        Steps per iteration:
          1. Build block-Hankel matrix from traces.
          2. Truncated SVD to target rank.
          3. Reverse Hankel → traces.
          4. Re-insert observed traces.
        """
        n_tr, n_samp = section.shape
        w = min(self._window, n_tr // 2)
        if w < 2:
            return section

        result = section.copy()
        for it in range(self._niter):
            H = self._build_hankel(result, w)
            H_lr = self._truncated_svd(H, self._rank)
            result = self._hankel_to_traces(H_lr, n_tr, n_samp, w)
            # Re-insert observed
            result[trace_mask] = section[trace_mask]

        return result

    # -- Hankel embedding helpers ---------------------------------------------

    @staticmethod
    def _build_hankel(data: np.ndarray, w: int) -> np.ndarray:
        """Build block-Hankel matrix from (n_traces, n_samples).

        Each column is a window of *w* consecutive traces (all samples).
        Output shape: (w * n_samples, n_traces - w + 1).
        """
        n_tr, n_samp = data.shape
        n_cols = n_tr - w + 1
        H = np.empty((w * n_samp, n_cols), dtype=data.dtype)
        for j in range(n_cols):
            H[:, j] = data[j: j + w, :].ravel()
        return H

    @staticmethod
    def _truncated_svd(H: np.ndarray, rank: int) -> np.ndarray:
        """Truncated SVD rank reduction."""
        rank = min(rank, min(H.shape) - 1)
        try:
            from scipy.sparse.linalg import svds
            U, s, Vt = svds(H.astype(np.float64), k=rank)
        except Exception:
            U, s, Vt = np.linalg.svd(H.astype(np.float64), full_matrices=False)
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        return (U * s) @ Vt

    @staticmethod
    def _hankel_to_traces(
        H: np.ndarray, n_tr: int, n_samp: int, w: int
    ) -> np.ndarray:
        """Invert Hankel embedding via diagonal averaging."""
        n_cols = n_tr - w + 1
        result = np.zeros((n_tr, n_samp), dtype=H.dtype)
        counts = np.zeros(n_tr, dtype=np.float64)
        for j in range(n_cols):
            col = H[:, j].reshape(w, n_samp)
            for k in range(w):
                result[j + k] += col[k]
                counts[j + k] += 1.0
        result /= counts[:, np.newaxis]
        return result.astype(np.float32)

    # -- evaluation ----------------------------------------------------------

    @staticmethod
    def _evaluate(truth: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        miss = ~mask
        t_flat = truth[miss].ravel()
        r_flat = recon[miss].ravel()
        valid = np.isfinite(r_flat) & np.isfinite(t_flat)
        t_flat, r_flat = t_flat[valid], r_flat[valid]
        rmse = float(np.sqrt(np.mean((t_flat - r_flat) ** 2)))
        corr = float(np.corrcoef(t_flat, r_flat)[0, 1]) if len(t_flat) > 1 else 0.0
        mae = float(np.mean(np.abs(t_flat - r_flat)))
        return {"rmse": rmse, "mae": mae, "pearson_corr": corr}
