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

        with segyio.create(str(self._path), spec) as f:
            # Binary header
            f.bin[segyio.BinField.Interval] = self._dt_us
            f.bin[segyio.BinField.Samples] = n_samp
            f.bin[segyio.BinField.Format] = self._sample_format
            f.bin[segyio.BinField.Traces] = n_il * n_xl

            trace_idx = 0
            for i_idx, il in enumerate(self._inlines):
                for x_idx, xl in enumerate(self._xlines):
                    f.trace[trace_idx] = volume[i_idx, x_idx, :].astype(np.float32)

                    h = f.header[trace_idx]
                    h[self._hb.inline] = int(il)
                    h[self._hb.xline] = int(xl)

                    # Compute world coordinates
                    wx = self._origin_x + i_idx * self._il_step_x + x_idx * self._xl_step_x
                    wy = self._origin_y + i_idx * self._il_step_y + x_idx * self._xl_step_y

                    # Apply scalar for storage
                    sx, sy = self._scale_coords(wx, wy)
                    h[segyio.TraceField.SourceGroupScalar] = self._coord_scalar
                    h[segyio.TraceField.CDP_X] = sx
                    h[segyio.TraceField.CDP_Y] = sy
                    h[segyio.TraceField.SourceX] = sx
                    h[segyio.TraceField.SourceY] = sy

                    trace_idx += 1

        logger.info("SEG-Y written: {}", self._path)
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
