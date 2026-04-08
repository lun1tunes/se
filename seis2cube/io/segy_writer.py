"""Write structured 3D SEG-Y files via segyio.

Supports segyio.create(spec) for full control over headers and
segyio.tools.from_array3D as a convenience shortcut.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio
from loguru import logger

from seis2cube.config import SegyHeaderBytes


class SegyWriter3D:
    """Writes a 3D numpy array as a structured SEG-Y file.

    Parameters
    ----------
    path : output file path
    inlines, xlines : 1-D arrays of inline/crossline labels
    dt_us : sample interval in microseconds
    header_bytes : byte locations for inline/xline
    origin_x, origin_y : real-world origin of grid corner
    il_step_x, il_step_y : coordinate increment per inline step
    xl_step_x, xl_step_y : coordinate increment per crossline step
    coord_scalar : SEG-Y scalar for coordinates (e.g. -100 → divide by 100)
    sample_format : SEG-Y format code (default 5 = IEEE float 4 byte)
    """

    def __init__(
        self,
        path: str | Path,
        inlines: np.ndarray,
        xlines: np.ndarray,
        dt_us: int,
        header_bytes: SegyHeaderBytes | None = None,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        il_step_x: float = 0.0,
        il_step_y: float = 0.0,
        xl_step_x: float = 0.0,
        xl_step_y: float = 0.0,
        coord_scalar: int = -100,
        sample_format: int = 5,
        delrt_ms: float = 0.0,
    ) -> None:
        self._path = Path(path)
        self._inlines = np.asarray(inlines, dtype=np.int32)
        self._xlines = np.asarray(xlines, dtype=np.int32)
        self._dt_us = dt_us
        self._hb = header_bytes or SegyHeaderBytes()
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._il_step_x = il_step_x
        self._il_step_y = il_step_y
        self._xl_step_x = xl_step_x
        self._xl_step_y = xl_step_y
        self._coord_scalar = coord_scalar
        self._sample_format = sample_format
        self._delrt_ms = delrt_ms

    def write(self, volume: np.ndarray) -> Path:
        """Write *volume* shaped (n_iline, n_xline, n_samples) to SEG-Y.

        Returns the output path.
        """
        n_il, n_xl, n_samp = volume.shape
        assert n_il == len(self._inlines), "inline dimension mismatch"
        assert n_xl == len(self._xlines), "xline dimension mismatch"

        self._path.parent.mkdir(parents=True, exist_ok=True)

        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
        spec.format = self._sample_format
        spec.iline = self._hb.inline
        spec.xline = self._hb.xline
        spec.samples = np.arange(n_samp, dtype=np.float32) * (self._dt_us / 1000.0)
        spec.ilines = self._inlines
        spec.xlines = self._xlines

        logger.info(
            "Writing SEG-Y {} — {}il × {}xl × {}samp",
            self._path.name, n_il, n_xl, n_samp,
        )

        import time as _time

        # Pre-compute all scaled coordinates (vectorised)
        ii = np.arange(n_il, dtype=np.float64)
        xi = np.arange(n_xl, dtype=np.float64)
        ii_grid, xi_grid = np.meshgrid(ii, xi, indexing="ij")
        wx_all = self._origin_x + ii_grid * self._il_step_x + xi_grid * self._xl_step_x
        wy_all = self._origin_y + ii_grid * self._il_step_y + xi_grid * self._xl_step_y
        sx_all, sy_all = self._scale_coords_array(wx_all.ravel(), wy_all.ravel())

        total_traces = n_il * n_xl
        log_every = max(1, total_traces // 10)
        t0 = _time.perf_counter()

        with segyio.create(str(self._path), spec) as f:
            # Binary header
            f.bin[segyio.BinField.Interval] = self._dt_us
            f.bin[segyio.BinField.Samples] = n_samp
            f.bin[segyio.BinField.Format] = self._sample_format
            f.bin[segyio.BinField.Traces] = total_traces

            trace_idx = 0
            for i_idx, il in enumerate(self._inlines):
                for x_idx, xl in enumerate(self._xlines):
                    f.trace[trace_idx] = volume[i_idx, x_idx, :].astype(np.float32)

                    h = f.header[trace_idx]
                    h[self._hb.inline] = int(il)
                    h[self._hb.xline] = int(xl)

                    h[segyio.TraceField.SourceGroupScalar] = self._coord_scalar
                    h[segyio.TraceField.CDP_X] = int(sx_all[trace_idx])
                    h[segyio.TraceField.CDP_Y] = int(sy_all[trace_idx])
                    h[segyio.TraceField.SourceX] = int(sx_all[trace_idx])
                    h[segyio.TraceField.SourceY] = int(sy_all[trace_idx])
                    h[segyio.TraceField.DelayRecordingTime] = int(self._delrt_ms)

                    trace_idx += 1

                if (trace_idx) % log_every < n_xl:
                    elapsed = _time.perf_counter() - t0
                    pct = 100 * trace_idx / total_traces
                    logger.info(
                        "Writing SEG-Y: {}/{} traces ({:.0f}%), {:.1f}s elapsed",
                        trace_idx, total_traces, pct, elapsed,
                    )

        elapsed = _time.perf_counter() - t0
        size_mb = self._path.stat().st_size / (1024 * 1024)
        logger.info("SEG-Y written: {} ({:.1f} MB in {:.2f}s)", self._path.name, size_mb, elapsed)
        return self._path

    @staticmethod
    def write_from_array(
        path: str | Path,
        volume: np.ndarray,
        dt_us: int = 2000,
        sample_format: int = 5,
    ) -> Path:
        """Quick shortcut using segyio.tools.from_array3D (default iline=189, xline=193)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        segyio.tools.from_array3D(str(path), volume.astype(np.float32), dt=dt_us, format=sample_format)
        logger.info("SEG-Y (from_array3D) written: {}", path)
        return path

    # ------------------------------------------------------------------

    def _scale_coords(self, x: float, y: float) -> tuple[int, int]:
        """Scale world coords for integer storage in trace headers."""
        s = self._coord_scalar
        if s == 0:
            s = 1
        if s < 0:
            factor = abs(s)
            return int(round(x * factor)), int(round(y * factor))
        else:
            return int(round(x / s)), int(round(y / s))

    def _scale_coords_array(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised coordinate scaling for batch header writing."""
        s = self._coord_scalar
        if s == 0:
            s = 1
        if s < 0:
            factor = abs(s)
            return np.rint(x * factor).astype(np.int64), np.rint(y * factor).astype(np.int64)
        else:
            return np.rint(x / s).astype(np.int64), np.rint(y / s).astype(np.int64)
