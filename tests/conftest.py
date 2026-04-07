"""Shared test fixtures — synthetic SEG-Y generation and config helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def make_3d_segy(
    path: Path,
    n_il: int = 10,
    n_xl: int = 15,
    n_samp: int = 100,
    dt_us: int = 2000,
    il_start: int = 100,
    xl_start: int = 200,
    il_step: int = 1,
    xl_step: int = 1,
) -> np.ndarray:
    """Create a small synthetic structured 3D SEG-Y and return the volume array."""
    rng = np.random.default_rng(42)
    volume = rng.standard_normal((n_il, n_xl, n_samp)).astype(np.float32)
    # Add a simple reflector
    volume[:, :, n_samp // 3] += 5.0
    volume[:, :, 2 * n_samp // 3] -= 3.0

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5  # IEEE float
    spec.iline = 189
    spec.xline = 193
    spec.samples = np.arange(n_samp, dtype=np.float32) * (dt_us / 1000.0)
    spec.ilines = np.arange(il_start, il_start + n_il * il_step, il_step)
    spec.xlines = np.arange(xl_start, xl_start + n_xl * xl_step, xl_step)

    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Samples] = n_samp
        f.bin[segyio.BinField.Format] = 5
        t = 0
        for i_idx, il in enumerate(spec.ilines):
            for x_idx, xl in enumerate(spec.xlines):
                f.trace[t] = volume[i_idx, x_idx, :]
                h = f.header[t]
                h[189] = int(il)
                h[193] = int(xl)
                # Simple coordinates: X = xl * 25, Y = il * 25
                h[segyio.TraceField.SourceGroupScalar] = -100
                cdpx = int(xl * 25 * 100)
                cdpy = int(il * 25 * 100)
                h[segyio.TraceField.CDP_X] = cdpx
                h[segyio.TraceField.CDP_Y] = cdpy
                h[segyio.TraceField.SourceX] = cdpx
                h[segyio.TraceField.SourceY] = cdpy
                t += 1

    return volume


def make_2d_segy(
    path: Path,
    n_traces: int = 20,
    n_samp: int = 100,
    dt_us: int = 2000,
    x_start: float = 5000.0,
    y_start: float = 2500.0,
    dx: float = 25.0,
    dy: float = 0.0,
    amplitude_scale: float = 1.2,
    time_shift_samples: int = 2,
) -> np.ndarray:
    """Create a synthetic 2D SEG-Y line that roughly overlaps with the 3D cube.

    Applies a known amplitude scale and time shift so calibration can be verified.
    """
    rng = np.random.default_rng(123)
    data = rng.standard_normal((n_traces, n_samp)).astype(np.float32)
    # Add reflectors at similar positions as the 3D
    data[:, n_samp // 3] += 5.0 * amplitude_scale
    data[:, 2 * n_samp // 3] -= 3.0 * amplitude_scale
    # Apply time shift
    if time_shift_samples != 0:
        data = np.roll(data, time_shift_samples, axis=1)

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5
    spec.samples = np.arange(n_samp, dtype=np.float32) * (dt_us / 1000.0)
    spec.tracecount = n_traces

    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Samples] = n_samp
        f.bin[segyio.BinField.Format] = 5
        for i in range(n_traces):
            f.trace[i] = data[i]
            h = f.header[i]
            h[segyio.TraceField.SourceGroupScalar] = -100
            x = x_start + i * dx
            y = y_start + i * dy
            h[segyio.TraceField.CDP_X] = int(x * 100)
            h[segyio.TraceField.CDP_Y] = int(y * 100)
            h[segyio.TraceField.SourceX] = int(x * 100)
            h[segyio.TraceField.SourceY] = int(y * 100)

    return data
