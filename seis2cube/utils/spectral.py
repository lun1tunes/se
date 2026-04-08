"""Spectral and signal-processing utilities for calibration / QC."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def analytic_signal(trace: np.ndarray) -> np.ndarray:
    """Return analytic (complex) signal via Hilbert transform."""
    return hilbert(trace, axis=-1)


def envelope(trace: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude (envelope)."""
    return np.abs(analytic_signal(trace))


def instantaneous_phase(trace: np.ndarray) -> np.ndarray:
    """Instantaneous phase in radians."""
    return np.angle(analytic_signal(trace))


def rms_amplitude(trace: np.ndarray, axis: int = -1) -> np.ndarray:
    """RMS amplitude along *axis*."""
    return np.sqrt(np.mean(trace ** 2, axis=axis))


def amplitude_spectrum(trace: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_hz, amplitude_spectrum) for a trace / array of traces.

    If trace is 2-D (n_traces, n_samples), spectra are computed per trace.
    """
    n = trace.shape[-1]
    dt_s = dt_ms / 1000.0
    freqs = np.fft.rfftfreq(n, d=dt_s)
    spec = np.abs(np.fft.rfft(trace, axis=-1))
    return freqs, spec


def phase_rotate(trace: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply constant phase rotation (in degrees) to a trace."""
    angle_rad = np.deg2rad(angle_deg)
    a = analytic_signal(trace)
    rotated = np.real(a * np.exp(1j * angle_rad))
    return rotated.astype(trace.dtype)


def matching_filter(
    trace_in: np.ndarray,
    trace_ref: np.ndarray,
    n_taps: int = 51,
) -> np.ndarray:
    """Design a Wiener-type matching filter that transforms trace_in → trace_ref.

    Uses least-squares in the frequency domain.
    Returns the filter coefficients.
    """
    from scipy.linalg import toeplitz

    n = len(trace_in)
    half = n_taps // 2
    # Auto-correlation of input
    rxx = np.correlate(trace_in, trace_in, mode="full")
    center = len(rxx) // 2
    r = rxx[center: center + n_taps]
    # Cross-correlation
    rxy = np.correlate(trace_ref, trace_in, mode="full")
    g = rxy[center - half: center - half + n_taps]
    # Toeplitz solve
    R = toeplitz(r)
    # Regularisation
    R += np.eye(n_taps) * 1e-6 * np.max(np.abs(R))
    filt = np.linalg.solve(R, g)
    return filt


def apply_matching_filter(trace: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Convolve trace with a matching filter (same-length output)."""
    from scipy.signal import fftconvolve
    out = fftconvolve(trace, filt, mode="same")
    return out.astype(trace.dtype)


def cross_correlation_shift(
    trace_a: np.ndarray, trace_b: np.ndarray, max_shift_samples: int | None = None
) -> tuple[int, float]:
    """Find the integer sample shift that maximizes cross-correlation.

    Returns (shift_samples, cc_max).  Positive shift means trace_a is delayed
    relative to trace_b.

    Uses FFT-based correlation — O(n log n) instead of O(n²).
    """
    from scipy.signal import fftconvolve
    # Cross-correlation via fftconvolve (flip b) — O(n log n)
    cc = fftconvolve(trace_a, trace_b[::-1], mode="full")
    center = len(cc) // 2
    if max_shift_samples is not None:
        lo = max(0, center - max_shift_samples)
        hi = min(len(cc), center + max_shift_samples + 1)
        cc_window = cc[lo:hi]
        best = np.argmax(np.abs(cc_window)) + lo
    else:
        best = np.argmax(np.abs(cc))
    shift = best - center
    cc_max = float(cc[best]) / max(np.sqrt(np.sum(trace_a ** 2) * np.sum(trace_b ** 2)), 1e-30)
    return int(shift), cc_max
