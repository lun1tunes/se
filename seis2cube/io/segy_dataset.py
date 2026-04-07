"""Unified SEG-Y reader abstraction over segyio.

Handles:
 - Configurable header byte positions (inline/xline/X/Y).
 - Multiple coordinate sources (Source, Group, CDP).
 - Coordinate scalar application per SEG-Y Rev 1 (bytes 71-72).
 - Lazy trace reading; optional mmap.
 - Both structured (geometry-aware) and unstructured modes.

SEG-Y Rev 1 reference:
  Binary header: sample interval µs in bytes 3217-3218, samples/trace 3221-3222,
  format code 3225-3226.  Trace header: coord scalar 71-72, SourceX/Y 73-80,
  GroupX/Y 81-88, coord units 89-90.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import segyio

from seis2cube.config import CoordSource, SegyHeaderBytes, IOConfig

# Mapping from CoordSource enum to segyio TraceField pairs (X, Y).
_COORD_FIELDS: dict[CoordSource, tuple[int, int]] = {
    CoordSource.SOURCE: (segyio.TraceField.SourceX, segyio.TraceField.SourceY),
    CoordSource.GROUP: (segyio.TraceField.GroupX, segyio.TraceField.GroupY),
    CoordSource.CDP: (segyio.TraceField.CDP_X, segyio.TraceField.CDP_Y),
}


@dataclass
class TraceHeader:
    """Lightweight container for a single trace's header values."""
    index: int
    inline: int | None
    xline: int | None
    x: float
    y: float
    delrt: float  # delay recording time (ms)


@dataclass
class SegyMeta:
    """Metadata extracted from a SEG-Y file."""
    path: Path
    n_traces: int
    n_samples: int
    sample_interval_us: int  # microseconds
    dt_ms: float  # milliseconds
    sample_format: int
    is_structured: bool
    inlines: np.ndarray | None = field(default=None)
    xlines: np.ndarray | None = field(default=None)


class SegyDataset:
    """Context-managed, lazy-reading abstraction over a SEG-Y file.

    Usage::

        with SegyDataset(path, header_bytes=hb, io_cfg=io) as ds:
            meta = ds.meta
            for hdr, data in ds.iter_traces():
                ...
    """

    def __init__(
        self,
        path: str | Path,
        header_bytes: SegyHeaderBytes | None = None,
        io_cfg: IOConfig | None = None,
        mode: str = "r",
    ) -> None:
        self._path = Path(path)
        self._hb = header_bytes or SegyHeaderBytes()
        self._io = io_cfg or IOConfig()
        self._mode = mode
        self._f: segyio.SegyFile | None = None
        self._meta: SegyMeta | None = None

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "SegyDataset":
        self.open()
        return self

    def __exit__(self, *exc) -> None:  # type: ignore[override]
        self.close()

    def open(self) -> None:
        kw: dict = {"mode": self._mode}
        if self._io.ignore_geometry:
            kw["ignore_geometry"] = True
        else:
            kw["iline"] = self._hb.inline
            kw["xline"] = self._hb.xline
            kw["strict"] = self._io.strict

        self._f = segyio.open(str(self._path), **kw)

        if self._io.mmap:
            ok = self._f.mmap()
            if not ok:
                from loguru import logger
                logger.warning("mmap() returned False for {}", self._path.name)

        self._meta = self._build_meta()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    # -- properties ----------------------------------------------------------

    @property
    def handle(self) -> segyio.SegyFile:
        assert self._f is not None, "SegyDataset not open"
        return self._f

    @property
    def meta(self) -> SegyMeta:
        assert self._meta is not None, "SegyDataset not open"
        return self._meta

    @property
    def n_traces(self) -> int:
        return self.meta.n_traces

    @property
    def n_samples(self) -> int:
        return self.meta.n_samples

    @property
    def dt_ms(self) -> float:
        return self.meta.dt_ms

    @property
    def time_axis_ms(self) -> np.ndarray:
        """Time axis in ms starting from 0."""
        return np.arange(self.n_samples) * self.dt_ms

    # -- coordinate helpers --------------------------------------------------

    def _apply_scalar(self, raw: int, scalar: int) -> float:
        """Apply SEG-Y coordinate scalar (bytes 71-72).

        If scalar > 0 → multiply; if scalar < 0 → divide by |scalar|;
        if scalar == 0 → treat as 1.
        """
        if scalar == 0:
            scalar = 1
        if scalar > 0:
            return float(raw * scalar)
        return float(raw) / abs(scalar)

    def _read_xy(self, trace_idx: int) -> tuple[float, float]:
        """Read (X, Y) for a given trace using configured CoordSource."""
        h = self._f.header[trace_idx]  # type: ignore[union-attr]
        fx, fy = _COORD_FIELDS[self._hb.coord_source]
        scalar = h.get(segyio.TraceField.SourceGroupScalar, 0)
        raw_x = h.get(fx, 0)
        raw_y = h.get(fy, 0)
        return self._apply_scalar(raw_x, scalar), self._apply_scalar(raw_y, scalar)

    def read_trace_header(self, idx: int) -> TraceHeader:
        h = self._f.header[idx]  # type: ignore[union-attr]
        iline = h.get(self._hb.inline, None) if not self._io.ignore_geometry else None
        xline = h.get(self._hb.xline, None) if not self._io.ignore_geometry else None
        x, y = self._read_xy(idx)
        delrt = float(h.get(segyio.TraceField.DelayRecordingTime, 0))
        return TraceHeader(index=idx, inline=iline, xline=xline, x=x, y=y, delrt=delrt)

    # -- bulk coordinate extraction ------------------------------------------

    def all_coordinates(self) -> np.ndarray:
        """Return (N, 2) array of (X, Y) for every trace."""
        fx, fy = _COORD_FIELDS[self._hb.coord_source]
        xs = np.array(self._f.attributes(fx)[:], dtype=np.float64)  # type: ignore[union-attr]
        ys = np.array(self._f.attributes(fy)[:], dtype=np.float64)  # type: ignore[union-attr]
        scalars = np.array(
            self._f.attributes(segyio.TraceField.SourceGroupScalar)[:],  # type: ignore[union-attr]
            dtype=np.float64,
        )
        # vectorised scalar application
        scalars[scalars == 0] = 1.0
        pos_mask = scalars > 0
        neg_mask = ~pos_mask
        xs[pos_mask] *= scalars[pos_mask]
        ys[pos_mask] *= scalars[pos_mask]
        xs[neg_mask] /= np.abs(scalars[neg_mask])
        ys[neg_mask] /= np.abs(scalars[neg_mask])
        return np.column_stack([xs, ys])

    def all_inlines_xlines(self) -> np.ndarray | None:
        """Return (N, 2) array of (inline, xline) or None if unstructured."""
        if self._io.ignore_geometry:
            return None
        ilines = np.array(
            self._f.attributes(self._hb.inline)[:], dtype=np.int32  # type: ignore[union-attr]
        )
        xlines = np.array(
            self._f.attributes(self._hb.xline)[:], dtype=np.int32  # type: ignore[union-attr]
        )
        return np.column_stack([ilines, xlines])

    # -- trace data access ---------------------------------------------------

    def read_trace(self, idx: int) -> np.ndarray:
        """Read a single trace (lazy, from disk)."""
        return np.array(self._f.trace[idx], dtype=np.float32)  # type: ignore[union-attr]

    def read_traces(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """Read multiple traces into (len(indices), n_samples) array."""
        out = np.empty((len(indices), self.n_samples), dtype=np.float32)
        for i, idx in enumerate(indices):
            out[i] = self._f.trace[int(idx)]  # type: ignore[union-attr]
        return out

    def iter_traces(self) -> Iterator[tuple[TraceHeader, np.ndarray]]:
        """Iterate lazily over (header, samples) for every trace."""
        for i in range(self.n_traces):
            yield self.read_trace_header(i), self.read_trace(i)

    def read_inline(self, il: int) -> np.ndarray:
        """Read a full inline gather (structured mode only)."""
        return np.array(self._f.iline[il], dtype=np.float32)  # type: ignore[union-attr]

    def read_xline(self, xl: int) -> np.ndarray:
        """Read a full crossline gather."""
        return np.array(self._f.xline[xl], dtype=np.float32)  # type: ignore[union-attr]

    def read_time_slice(self, sample_idx: int) -> np.ndarray:
        """Read a depth/time slice at given sample index (structured mode)."""
        # segyio provides f.depth_slice for structured files
        return np.array(
            self._f.depth_slice[sample_idx], dtype=np.float32  # type: ignore[union-attr]
        )

    # -- internal ------------------------------------------------------------

    def _build_meta(self) -> SegyMeta:
        f = self._f
        assert f is not None
        n_traces = f.tracecount
        n_samples = len(f.samples)
        # sample interval from binary header (bytes 3217-3218 → segyio key)
        si_us = int(f.bin[segyio.BinField.Interval])
        if si_us <= 0:
            from loguru import logger
            logger.warning("Sample interval in binary header is {} µs; defaulting to 2000 µs (2 ms)", si_us)
            si_us = 2000
        dt_ms = si_us / 1000.0
        fmt = int(f.bin[segyio.BinField.Format])

        is_structured = not self._io.ignore_geometry and hasattr(f, "ilines") and f.ilines is not None

        ilines = np.array(f.ilines) if is_structured else None
        xlines = np.array(f.xlines) if is_structured else None

        return SegyMeta(
            path=self._path,
            n_traces=n_traces,
            n_samples=n_samples,
            sample_interval_us=si_us,
            dt_ms=dt_ms,
            sample_format=fmt,
            is_structured=is_structured,
            inlines=ilines,
            xlines=xlines,
        )
