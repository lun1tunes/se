"""Utility functions for array manipulation, windowing, and feature extraction."""

from __future__ import annotations

import numpy as np


def sliding_windows(
    n_samples: int,
    window_ms: float,
    overlap_ms: float,
    dt_ms: float,
) -> list[tuple[int, int]]:
    """Return list of (start_sample, end_sample) tuples for sliding windows."""
    win_samp = max(1, int(round(window_ms / dt_ms)))
    step_samp = max(1, win_samp - int(round(overlap_ms / dt_ms)))
    windows = []
    start = 0
    while start < n_samples:
        end = min(start + win_samp, n_samples)
        windows.append((start, end))
        start += step_samp
        if end == n_samples:
            break
    return windows


def cosine_taper(n: int) -> np.ndarray:
    """1-D cosine taper from 0 → 1 over *n* samples."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    return (0.5 * (1.0 - np.cos(np.pi * np.arange(n) / max(n - 1, 1)))).astype(np.float32)


def blend_boundary(
    cube_orig: np.ndarray,
    cube_ext: np.ndarray,
    mask_orig: np.ndarray,
    taper_width: int = 10,
) -> np.ndarray:
    """Blend original 3D and extended 3D at the boundary via cosine taper.

    Parameters
    ----------
    cube_orig : (n_il, n_xl, n_samp) — original 3D data.
    cube_ext : same shape — reconstructed extension.
    mask_orig : (n_il, n_xl) bool — True where original 3D data exists.
    taper_width : number of traces for cosine taper at the boundary.

    Returns
    -------
    blended : merged volume.
    """
    from scipy.ndimage import distance_transform_edt

    # Distance from mask boundary (in grid cells)
    dist_inside = distance_transform_edt(mask_orig)
    dist_outside = distance_transform_edt(~mask_orig)

    # Normalised blend weight: 1 inside original, 0 far outside, smooth at boundary
    weight = np.clip(dist_inside / max(taper_width, 1), 0.0, 1.0)

    # Expand weight to 3D
    w3 = weight[:, :, np.newaxis].astype(np.float32)
    blended = cube_orig * w3 + cube_ext * (1.0 - w3)
    return blended


def extract_features(trace: np.ndarray, dt_ms: float) -> dict[str, float]:
    """Extract simple per-trace features for regression-based calibration."""
    from seis2cube.utils.spectral import envelope, rms_amplitude

    env = envelope(trace)
    return {
        "amplitude_mean": float(np.mean(trace)),
        "amplitude_std": float(np.std(trace)),
        "rms": float(rms_amplitude(trace)),
        "envelope_mean": float(np.mean(env)),
        "envelope_max": float(np.max(env)),
        "d_dt_rms": float(rms_amplitude(np.diff(trace))),
    }


def extract_window_features(
    trace: np.ndarray,
    start: int,
    end: int,
    dt_ms: float,
) -> dict[str, float]:
    """Extract features for a time-window of a trace."""
    window = trace[start:end]
    feats = extract_features(window, dt_ms)
    feats["window_start_ms"] = float(start * dt_ms)
    feats["window_end_ms"] = float(end * dt_ms)
    return feats
