"""Level-0 calibration: global constant corrections.

Estimates per-dataset (or per-line):
  - Δt  — bulk time shift (integer samples via cross-correlation).
  - g   — scalar gain (amplitude ratio).
  - φ   — constant phase rotation (degrees).
  - (optional) spectral matching filter.
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
from seis2cube.utils.spectral import (
    cross_correlation_shift,
    matching_filter,
    apply_matching_filter,
    phase_rotate,
)


class GlobalShiftGainPhase(CalibrationStrategy):
    """Constant Δt / gain / phase rotation calibration.

    Parameters
    ----------
    max_shift_ms : maximum allowed time shift in ms.
    estimate_phase : whether to estimate and apply phase rotation.
    estimate_matching_filter : whether to compute a spectral matching filter.
    matching_filter_taps : number of filter taps (odd).
    """

    def __init__(
        self,
        max_shift_ms: float = 50.0,
        estimate_phase: bool = True,
        estimate_matching_filter: bool = False,
        matching_filter_taps: int = 51,
    ) -> None:
        self._max_shift_ms = max_shift_ms
        self._est_phase = estimate_phase
        self._est_mf = estimate_matching_filter
        self._mf_taps = matching_filter_taps

    def fit(self, pairs: CalibrationPair) -> CalibrationModel:
        dt = pairs.dt_ms
        max_shift_samp = max(1, int(round(self._max_shift_ms / dt)))

        # Stack traces to get a robust estimate
        mean_2d = pairs.amp_2d.mean(axis=0)
        mean_3d = pairs.amp_3d.mean(axis=0)

        # 1. Time shift via cross-correlation
        shift_samp, cc = cross_correlation_shift(mean_3d, mean_2d, max_shift_samp)
        shift_ms = shift_samp * dt
        logger.info("Global Δt = {:.1f} ms ({} samples), CC={:.3f}", shift_ms, shift_samp, cc)

        # Apply shift to 2D for further estimates
        shifted_2d = np.roll(mean_2d, shift_samp)

        # 2. Gain (RMS ratio)
        rms_3d = np.sqrt(np.mean(mean_3d ** 2))
        rms_2d = np.sqrt(np.mean(shifted_2d ** 2))
        gain = rms_3d / max(rms_2d, 1e-30)
        logger.info("Global gain = {:.4f}", gain)

        # 3. Phase rotation (optional) — grid search
        phase_deg = 0.0
        if self._est_phase:
            best_cc = -1.0
            for phi in np.arange(-180, 180, 5):
                rotated = phase_rotate(shifted_2d * gain, phi)
                c = float(np.corrcoef(rotated, mean_3d)[0, 1])
                if c > best_cc:
                    best_cc = c
                    phase_deg = phi
            logger.info("Global phase = {:.0f}°, CC={:.3f}", phase_deg, best_cc)

        # 4. Matching filter (optional)
        mf_coeffs: list[float] | None = None
        if self._est_mf:
            corrected_mean = phase_rotate(shifted_2d * gain, phase_deg)
            mf = matching_filter(corrected_mean, mean_3d, self._mf_taps)
            mf_coeffs = mf.tolist()
            logger.info("Matching filter estimated ({} taps)", len(mf))

        return CalibrationModel(
            method="global_shift_gain_phase",
            params={
                "shift_samples": int(shift_samp),
                "shift_ms": float(shift_ms),
                "gain": float(gain),
                "phase_deg": float(phase_deg),
                "matching_filter": mf_coeffs,
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
        out = amp_2d.copy()

        # 1. Time shift (vectorised over all traces)
        shift = p["shift_samples"]
        if shift != 0:
            out = np.roll(out, shift, axis=-1)

        # 2. Gain (vectorised)
        out *= p["gain"]

        # 3. Phase rotation — batch via scipy.signal.hilbert on full 2D array
        if p["phase_deg"] != 0.0:
            from scipy.signal import hilbert as _hilbert
            angle_rad = np.deg2rad(p["phase_deg"])
            analytic = _hilbert(out, axis=-1)
            out = np.real(analytic * np.exp(1j * angle_rad)).astype(np.float32)

        # 4. Matching filter — batch via fftconvolve (supports N-D)
        if p.get("matching_filter") is not None:
            from scipy.signal import fftconvolve
            mf = np.array(p["matching_filter"], dtype=np.float32)
            # fftconvolve along last axis for each trace
            out = fftconvolve(out, mf[np.newaxis, :], mode="same", axes=-1).astype(np.float32)

        result = out.astype(np.float32)
        return result[0] if was_1d else result
