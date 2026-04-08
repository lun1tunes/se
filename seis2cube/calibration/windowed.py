"""Level-1 calibration: windowed (time-varying) shift + gain.

Estimates Δt(t) and g(t) in overlapping time windows, then smoothly
interpolates across the full trace.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from seis2cube.calibration.base import (
    CalibrationModel,
    CalibrationPair,
    CalibrationStrategy,
)
from seis2cube.models.line2d import Line2D
from seis2cube.utils.array_utils import sliding_windows
from seis2cube.utils.spectral import cross_correlation_shift


class WindowedShiftGain(CalibrationStrategy):
    """Time-varying Δt and gain estimated in sliding windows.

    Parameters
    ----------
    window_ms : window length in ms.
    overlap_ms : overlap in ms.
    max_shift_ms : per-window maximum shift.
    cc_threshold : minimum |CC| to accept a window estimate.
    """

    def __init__(
        self,
        window_ms: float = 400.0,
        overlap_ms: float = 100.0,
        max_shift_ms: float = 30.0,
        cc_threshold: float = 0.3,
    ) -> None:
        self._win_ms = window_ms
        self._ovlp_ms = overlap_ms
        self._max_shift_ms = max_shift_ms
        self._cc_thr = cc_threshold

    def fit(self, pairs: CalibrationPair) -> CalibrationModel:
        dt = pairs.dt_ms
        n_samples = pairs.amp_2d.shape[1]
        windows = sliding_windows(n_samples, self._win_ms, self._ovlp_ms, dt)
        max_shift_samp = max(1, int(round(self._max_shift_ms / dt)))

        # Average over traces for robust per-window estimate
        mean_2d = pairs.amp_2d.mean(axis=0)
        mean_3d = pairs.amp_3d.mean(axis=0)

        win_centers: list[float] = []
        shifts: list[int] = []
        gains: list[float] = []

        for s, e in windows:
            w2d = mean_2d[s:e]
            w3d = mean_3d[s:e]
            if np.std(w2d) < 1e-10 or np.std(w3d) < 1e-10:
                continue

            shift_s, cc = cross_correlation_shift(w3d, w2d, max_shift_samp)
            if abs(cc) < self._cc_thr:
                continue

            rms3 = np.sqrt(np.mean(w3d ** 2))
            rms2 = np.sqrt(np.mean(w2d ** 2))
            g = rms3 / max(rms2, 1e-30)

            center_ms = (s + e) / 2.0 * dt
            win_centers.append(center_ms)
            shifts.append(shift_s)
            gains.append(g)

        logger.info("Windowed calibration: {} / {} windows accepted", len(win_centers), len(windows))

        return CalibrationModel(
            method="windowed_shift_gain",
            params={
                "window_centers_ms": win_centers,
                "shifts_samples": shifts,
                "gains": gains,
                "dt_ms": dt,
                "n_samples": n_samples,
            },
        )

    def apply(self, line: Line2D, model: CalibrationModel) -> Line2D:
        corrected = self._apply_array(line.data, model)
        return Line2D(
            name=line.name,
            path=line.path,
            coords=line.coords.copy(),
            data=corrected,
            dt_ms=line.dt_ms,
            delrt_ms=line.delrt_ms,
            quality_flags=line.quality_flags,
        )

    def _apply_array(self, amp_2d: np.ndarray, model: CalibrationModel) -> np.ndarray:
        p = model.params
        was_1d = amp_2d.ndim == 1
        if was_1d:
            amp_2d = amp_2d[np.newaxis, :]

        n_samp = p["n_samples"]
        dt = p["dt_ms"]
        centers = np.array(p["window_centers_ms"])
        shifts_s = np.array(p["shifts_samples"], dtype=np.float64)
        gains_arr = np.array(p["gains"], dtype=np.float64)

        if len(centers) == 0:
            logger.warning("No accepted windows — returning unmodified data")
            return amp_2d[0].copy() if was_1d else amp_2d.copy()

        # Interpolate shift and gain to every sample
        time_axis = np.arange(n_samp) * dt
        shift_interp = np.interp(time_axis, centers, shifts_s)
        gain_interp = np.interp(time_axis, centers, gains_arr)

        # Vectorised: apply time-varying shift to all traces at once
        # using scipy.ndimage.map_coordinates (batch-capable)
        from scipy.ndimage import map_coordinates

        n_traces = amp_2d.shape[0]
        src_indices = np.arange(n_samp, dtype=np.float64) - shift_interp  # (n_samp,)

        # map_coordinates expects (n_dims, n_points) coordinate arrays
        # For each trace i, we sample at (i, src_indices[j]) for all j
        row_coords = np.repeat(np.arange(n_traces, dtype=np.float64), n_samp)
        col_coords = np.tile(src_indices, n_traces)
        coords = np.array([row_coords, col_coords])

        out = map_coordinates(
            amp_2d.astype(np.float64), coords, order=1, mode='constant', cval=0.0,
        ).reshape(n_traces, n_samp)

        # Apply time-varying gain (broadcast over all traces)
        out *= gain_interp[np.newaxis, :]

        result = out.astype(np.float32)
        return result[0] if was_1d else result
