"""POCS (Projection Onto Convex Sets) interpolation for seismic reconstruction.

Iteratively applies:
  1. Forward sparsifying transform (FFT / wavelet).
  2. Thresholding in transform domain (soft or hard).
  3. Inverse transform.
  4. Re-insert known observations (mask projection).

Supports regular POCS and fast POCS (FPOCS) with exponential/linear threshold
schedules.  Parallelisation over 2D time-slices is supported via Dask.

References:
  - POCS for seismic interpolation: widely used for irregular missing trace
    reconstruction with sampling matrix / mask approach.
  - FPOCS gives faster convergence via accelerated threshold decay.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from seis2cube.interpolation.base import InterpolationResult, InterpolationStrategy
from seis2cube.models.volume import SparseVolume


class POCSInterpolator(InterpolationStrategy):
    """POCS / FPOCS 3D seismic interpolator.

    Parameters
    ----------
    n_iter : number of POCS iterations.
    transform : 'fft' or 'wavelet'.
    fast : use FPOCS (faster threshold decay).
    threshold_schedule : 'exponential' or 'linear'.
    threshold_start_pct : starting threshold as percentile of transform coefficients.
    threshold_end_pct : ending threshold percentile.
    """

    def __init__(
        self,
        n_iter: int = 100,
        transform: str = "fft",
        fast: bool = True,
        threshold_schedule: str = "exponential",
        threshold_start_pct: float = 99.0,
        threshold_end_pct: float = 1.0,
    ) -> None:
        if transform not in ("fft",):
            raise ValueError(
                f"transform='{transform}' is not supported. "
                f"Only 'fft' is currently implemented. Wavelet support is planned."
            )
        self._niter = n_iter
        self._transform = transform
        self._fast = fast
        self._sched = threshold_schedule
        self._thr_start = threshold_start_pct
        self._thr_end = threshold_end_pct

    # -- Strategy interface --------------------------------------------------

    def fit(
        self,
        full_volume: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float]:
        """Simulate POCS on the 3D volume with the given observation mask."""
        sparse_data = full_volume.copy().astype(np.float32)
        sparse_data[~mask] = 0.0
        recon = self._pocs_3d(sparse_data, mask)
        return self._evaluate(full_volume, recon, mask)

    def reconstruct(self, sparse: SparseVolume) -> InterpolationResult:
        data = np.nan_to_num(sparse.data, nan=0.0).astype(np.float32)
        costs: list[float] = []
        recon = self._pocs_3d(data, sparse.mask, _cost_out=costs)
        return InterpolationResult(volume=recon, cost_history=costs)

    # -- core POCS -----------------------------------------------------------

    def _pocs_3d(
        self, data: np.ndarray, mask: np.ndarray, *, _cost_out: list | None = None,
    ) -> np.ndarray:
        """Run POCS on a 3D volume (il, xl, samp).

        Operates on 2D spatial slices (il, xl) for each time sample.
        Uses rFFT for real-valued seismic data (~2× faster than full FFT).
        """
        import time as _time

        n_il, n_xl, n_samp = data.shape
        result = data.copy()

        thresholds = self._threshold_schedule(self._niter)

        # Pre-compute observation indices for fast re-insertion
        obs_il, obs_xl = np.where(mask)
        obs_data = data[obs_il, obs_xl, :]  # (n_obs, n_samp)
        spatial_shape = (n_il, n_xl)
        miss = ~mask  # pre-compute loop-invariant missing mask

        logger.info(
            "POCS: {} iter, transform={}, fast={}, schedule={}, volume={}x{}x{}",
            self._niter, self._transform, self._fast, self._sched,
            n_il, n_xl, n_samp,
        )
        log_every = max(1, self._niter // 10)
        t_start = _time.perf_counter()

        for it in range(self._niter):
            thr = thresholds[it]

            # Forward transform — rFFT along spatial axes for real data
            if self._transform == "fft":
                coeffs = np.fft.rfftn(result, axes=(0, 1))
            else:
                coeffs = self._wavelet_forward(result)

            # Thresholding (soft)
            self._soft_threshold_inplace(coeffs, thr)

            # Inverse transform
            if self._transform == "fft":
                result = np.fft.irfftn(coeffs, s=spatial_shape, axes=(0, 1)).astype(np.float32)
            else:
                result = self._wavelet_inverse(coeffs)

            # Re-insert observations (indexed, avoids broadcast_to allocation)
            result[obs_il, obs_xl, :] = obs_data

            # Track convergence cost
            if _cost_out is not None:
                cost = float(np.mean(result[miss, :] ** 2))
                _cost_out.append(cost)

            if (it + 1) % log_every == 0:
                elapsed = _time.perf_counter() - t_start
                logger.info(
                    "POCS iter {}/{} ({:.0f}%), thr={:.4f}, elapsed={:.1f}s",
                    it + 1, self._niter, 100 * (it + 1) / self._niter, thr, elapsed,
                )

        total = _time.perf_counter() - t_start
        logger.info("POCS completed {} iterations in {:.2f}s ({:.3f}s/iter)",
                    self._niter, total, total / max(self._niter, 1))
        return result

    def _threshold_schedule(self, n: int) -> np.ndarray:
        """Generate threshold values for each iteration."""
        t_start = max(self._thr_start, 1e-6)
        t_end = max(self._thr_end, 1e-6)
        if self._sched == "exponential":
            if self._fast:
                # FPOCS: accelerated decay — use power-law with exponent > 1
                t = np.linspace(0.0, 1.0, n)
                alpha = 3.0  # acceleration factor
                decay = (1.0 - t) ** alpha
                return t_end + (t_start - t_end) * decay
            return np.logspace(np.log10(t_start), np.log10(t_end), n)
        # linear
        return np.linspace(t_start, t_end, n)

    @staticmethod
    def _soft_threshold_inplace(coeffs: np.ndarray, thr_pct: float) -> None:
        """In-place soft thresholding — avoids large temporary allocations."""
        abs_c = np.abs(coeffs)
        # Use partial sort (O(n)) instead of full sort percentile for large arrays
        flat = abs_c.ravel()
        k = int(thr_pct / 100.0 * len(flat))
        if k <= 0 or k >= len(flat):
            return
        thr_val = np.partition(flat, k)[k]  # O(n) vs O(n log n) for percentile
        # In-place shrinkage: coeffs *= max(|c| - thr, 0) / |c|
        # This preserves phase for complex, sign for real
        np.clip(abs_c, 1e-30, None, out=abs_c)  # avoid div-by-zero
        scale = np.maximum(abs_c - thr_val, 0.0)
        scale /= abs_c
        coeffs *= scale

    # -- wavelet stubs (require pywt) ----------------------------------------

    @staticmethod
    def _wavelet_forward(data: np.ndarray) -> np.ndarray:
        """2D DWT per time sample using PyWavelets.

        Raises NotImplementedError — wavelet round-trip is not yet implemented.
        Use transform='fft' instead.
        """
        raise NotImplementedError(
            "Wavelet transform is not implemented. Use transform='fft' (or 'fpocs') instead. "
            "Set transform='fft' in your InterpolationConfig."
        )

    @staticmethod
    def _wavelet_inverse(coeffs: np.ndarray) -> np.ndarray:
        """Inverse 2D DWT — not implemented."""
        raise NotImplementedError("Wavelet inverse is not implemented.")

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
