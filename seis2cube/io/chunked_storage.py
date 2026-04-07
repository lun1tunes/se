"""Optional adapter for out-of-core / chunked storage (Zarr, NetCDF, memmap).

Provides conversion between SEG-Y ↔ chunked formats and lazy array handles
for large-volume processing with Dask or numpy.memmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from seis2cube.config import ChunkingConfig


class ChunkedStorageAdapter:
    """Abstraction for reading/writing seismic volumes in chunked formats.

    Supports:
      - numpy.memmap  (always available)
      - zarr           (if installed)
      - dask.array     (if installed)
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._cfg = config or ChunkingConfig()

    # -- numpy memmap --------------------------------------------------------

    @staticmethod
    def create_memmap(
        path: str | Path,
        shape: tuple[int, ...],
        dtype: np.dtype | str = "float32",
        mode: str = "w+",
    ) -> np.memmap:
        """Create (or open) a memory-mapped numpy array on disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)
        logger.debug("memmap created: {} shape={} dtype={}", path, shape, dtype)
        return mm

    @staticmethod
    def open_memmap(
        path: str | Path,
        shape: tuple[int, ...],
        dtype: np.dtype | str = "float32",
    ) -> np.memmap:
        """Open an existing memory-mapped array in read-only mode."""
        return np.memmap(str(path), dtype=dtype, mode="r", shape=shape)

    # -- Zarr ----------------------------------------------------------------

    def to_zarr(
        self,
        data: np.ndarray | Any,
        path: str | Path,
        chunks: tuple[int, ...] | None = None,
    ) -> Any:
        """Write a volume to Zarr store.  Returns zarr.Array."""
        import zarr  # optional dep

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if chunks is None:
            chunks = (
                self._cfg.inline_chunk,
                self._cfg.xline_chunk,
                self._cfg.time_chunk,
            )
        store = zarr.open(
            str(path), mode="w", shape=data.shape, dtype=data.dtype, chunks=chunks
        )
        store[:] = data
        logger.info("Zarr written: {} chunks={}", path, chunks)
        return store

    @staticmethod
    def from_zarr(path: str | Path) -> Any:
        """Open a Zarr store lazily."""
        import zarr
        return zarr.open(str(path), mode="r")

    # -- Dask convenience ----------------------------------------------------

    def as_dask_array(
        self,
        data: np.ndarray | Any,
        chunks: tuple[int, ...] | None = None,
    ) -> Any:
        """Wrap data as a dask.array with configured chunk sizes."""
        import dask.array as da  # optional dep

        if chunks is None:
            chunks = (
                self._cfg.inline_chunk,
                self._cfg.xline_chunk,
                self._cfg.time_chunk,
            )
        if isinstance(data, np.ndarray):
            return da.from_array(data, chunks=chunks)
        return da.from_array(data, chunks=chunks)
