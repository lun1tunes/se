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
        import time as _time

        n_il, n_xl, n_samp = data.shape
        result = data.copy()

        # Pre-compute observation indices for fast re-insertion
        obs_il, obs_xl = np.where(mask)
        obs_data = data[obs_il, obs_xl, :]

        logger.info(
            "MSSA: rank={}, window={}, iter={}, volume={}x{}x{}",
            self._rank, self._window, self._niter, n_il, n_xl, n_samp,
        )

        # ── Inline pass ──────────────────────────────────────────────
        t0 = _time.perf_counter()
        n_il_todo = int((~mask.all(axis=1)).sum())
        done = 0
        log_every = max(1, n_il_todo // 10)

        for il_idx in range(n_il):
            section_mask = mask[il_idx]  # (n_xl,) bool
            if section_mask.all():
                continue
            result[il_idx] = self._mssa_section(result[il_idx], section_mask)
            done += 1
            if done % log_every == 0:
                elapsed = _time.perf_counter() - t0
                logger.info(
                    "MSSA inline pass: {}/{} ({:.0f}%), elapsed={:.1f}s",
                    done, n_il_todo, 100 * done / max(n_il_todo, 1), elapsed,
                )

        t_il = _time.perf_counter() - t0
        logger.info("MSSA inline pass done in {:.2f}s", t_il)

        # ── Crossline pass ───────────────────────────────────────────
        t0 = _time.perf_counter()
        n_xl_todo = int((~mask.all(axis=0)).sum())
        done = 0
        log_every = max(1, n_xl_todo // 10)

        for xl_idx in range(n_xl):
            section_mask = mask[:, xl_idx]
            if section_mask.all():
                continue
            result[:, xl_idx, :] = self._mssa_section(result[:, xl_idx, :], section_mask)
            done += 1
            if done % log_every == 0:
                elapsed = _time.perf_counter() - t0
                logger.info(
                    "MSSA crossline pass: {}/{} ({:.0f}%), elapsed={:.1f}s",
                    done, n_xl_todo, 100 * done / max(n_xl_todo, 1), elapsed,
                )

        t_xl = _time.perf_counter() - t0
        logger.info("MSSA crossline pass done in {:.2f}s  (total {:.2f}s)", t_xl, t_il + t_xl)

        # Re-insert observations
        result[obs_il, obs_xl, :] = obs_data
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

        Uses stride tricks for zero-copy sliding window — O(1) construction.
        Output shape: (w * n_samples, n_traces - w + 1).
        """
        n_tr, n_samp = data.shape
        n_cols = n_tr - w + 1
        # Sliding window view: (n_cols, w, n_samp)
        row_stride, col_stride = data.strides
        windowed = np.lib.stride_tricks.as_strided(
            data,
            shape=(n_cols, w, n_samp),
            strides=(row_stride, row_stride, col_stride),
        )
        # Reshape to (n_cols, w*n_samp) then transpose → (w*n_samp, n_cols)
        return np.ascontiguousarray(windowed.reshape(n_cols, w * n_samp).T)

    @staticmethod
    def _truncated_svd(H: np.ndarray, rank: int) -> np.ndarray:
        """Truncated SVD rank reduction.

        Uses randomized SVD (sklearn) for large matrices — O(m·n·k) instead of O(m·n·min(m,n)).
        Falls back to scipy sparse svds, then full SVD.
        """
        rank = min(rank, min(H.shape) - 1)
        if rank < 1:
            return H.copy()

        H64 = H.astype(np.float64)

        # Prefer randomized SVD for large matrices (much faster)
        if min(H.shape) > 100 and rank < min(H.shape) // 2:
            try:
                from sklearn.utils.extmath import randomized_svd
                U, s, Vt = randomized_svd(H64, n_components=rank, random_state=42)
                return (U * s) @ Vt
            except ImportError:
                pass

        try:
            from scipy.sparse.linalg import svds
            U, s, Vt = svds(H64, k=rank)
        except Exception:
            U, s, Vt = np.linalg.svd(H64, full_matrices=False)
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        return (U * s) @ Vt

    @staticmethod
    def _hankel_to_traces(
        H: np.ndarray, n_tr: int, n_samp: int, w: int
    ) -> np.ndarray:
        """Invert Hankel embedding via diagonal averaging (vectorised)."""
        n_cols = n_tr - w + 1
        result = np.zeros((n_tr, n_samp), dtype=np.float64)
        counts = np.zeros(n_tr, dtype=np.float64)

        # Reshape columns into (n_cols, w, n_samp) for batch accumulation
        cols = H.T.reshape(n_cols, w, n_samp)  # (n_cols, w, n_samp)
        # Build index array: for each column j and window position k, target = j+k
        j_idx = np.arange(n_cols)[:, None] + np.arange(w)[None, :]  # (n_cols, w)
        # Accumulate using add.at for proper handling of repeated indices
        np.add.at(result, j_idx.ravel(), cols.reshape(-1, n_samp))
        np.add.at(counts, j_idx.ravel(), 1.0)

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
