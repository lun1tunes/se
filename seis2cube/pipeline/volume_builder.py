"""VolumeBuilder: constructs the target 3D grid and assembles the final cube.

Responsibilities:
  - Define extended grid (inlines/xlines) covering original 3D + expansion polygon.
  - Map calibrated 2D traces into the sparse grid.
  - Merge original 3D with reconstructed extension (with optional seam blending).
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from shapely.geometry import Polygon

from seis2cube.geometry.geometry_model import GeometryModel3D
from seis2cube.models.line2d import Line2D
from seis2cube.models.volume import SparseVolume, TargetGrid
from seis2cube.utils.array_utils import blend_boundary


class VolumeBuilder:
    """Builds the target 3D grid and populates it with data.

    Parameters
    ----------
    geometry : GeometryModel3D of the original 3D cube.
    orig_inlines, orig_xlines : inline/xline arrays of the original 3D.
    n_samples : number of time samples.
    dt_ms : sample interval in ms.
    expand_polygon : Shapely polygon defining the expansion area.
    grid_il_step : inline step override (None → use original step).
    grid_xl_step : xline step override (None → use original step).
    """

    def __init__(
        self,
        geometry: GeometryModel3D,
        orig_inlines: np.ndarray,
        orig_xlines: np.ndarray,
        n_samples: int,
        dt_ms: float,
        expand_polygon: Polygon,
        grid_il_step: float | None = None,
        grid_xl_step: float | None = None,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        il_step_x: float = 0.0,
        il_step_y: float = 0.0,
        xl_step_x: float = 0.0,
        xl_step_y: float = 0.0,
    ) -> None:
        self._geom = geometry
        self._orig_il = np.asarray(orig_inlines)
        self._orig_xl = np.asarray(orig_xlines)
        self._n_samp = n_samples
        self._dt_ms = dt_ms
        self._expand_poly = expand_polygon

        # Grid steps
        if grid_il_step is not None:
            self._dil = grid_il_step
        elif len(self._orig_il) > 1:
            self._dil = float(self._orig_il[1] - self._orig_il[0])
        else:
            self._dil = 1.0

        if grid_xl_step is not None:
            self._dxl = grid_xl_step
        elif len(self._orig_xl) > 1:
            self._dxl = float(self._orig_xl[1] - self._orig_xl[0])
        else:
            self._dxl = 1.0

        # Affine params (passed through from original 3D geometry estimation)
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._il_step_x = il_step_x
        self._il_step_y = il_step_y
        self._xl_step_x = xl_step_x
        self._xl_step_y = xl_step_y

    def build_target_grid(self, max_volume_gb: float = 0.0) -> TargetGrid:
        """Compute extended inline/xline ranges covering both the original 3D and the polygon."""
        # Get polygon bounding box in world coordinates
        minx, miny, maxx, maxy = self._expand_poly.bounds

        # Convert bounding box corners to fractional inline/xline
        corners_x = np.array([minx, maxx, minx, maxx])
        corners_y = np.array([miny, miny, maxy, maxy])
        il_frac, xl_frac = self._geom.xy_to_ilxl(corners_x, corners_y)

        # Extended range
        il_min = min(self._orig_il.min(), np.floor(il_frac.min()).astype(int))
        il_max = max(self._orig_il.max(), np.ceil(il_frac.max()).astype(int))
        xl_min = min(self._orig_xl.min(), np.floor(xl_frac.min()).astype(int))
        xl_max = max(self._orig_xl.max(), np.ceil(xl_frac.max()).astype(int))

        # Snap to step
        dil = int(self._dil)
        dxl = int(self._dxl)
        il_min = int(np.floor(il_min / dil) * dil)
        il_max = int(np.ceil(il_max / dil) * dil)
        xl_min = int(np.floor(xl_min / dxl) * dxl)
        xl_max = int(np.ceil(xl_max / dxl) * dxl)

        inlines = np.arange(il_min, il_max + 1, dil, dtype=np.int32)
        xlines = np.arange(xl_min, xl_max + 1, dxl, dtype=np.int32)

        # ── Memory safety: shrink extension range if grid exceeds limit ──
        if max_volume_gb > 0:
            vol_bytes = len(inlines) * len(xlines) * self._n_samp * 4
            vol_gb = vol_bytes / (1024 ** 3)
            if vol_gb > max_volume_gb:
                # Shrink the extension symmetrically while keeping
                # original IL/XL fully inside. We reduce the total range
                # until the volume fits.
                scale = (max_volume_gb / vol_gb) ** 0.5
                n_il_target = max(int(len(inlines) * scale), len(self._orig_il))
                n_xl_target = max(int(len(xlines) * scale), len(self._orig_xl))
                # Centre the new range on the original 3D centre
                il_ctr = (self._orig_il.min() + self._orig_il.max()) / 2.0
                xl_ctr = (self._orig_xl.min() + self._orig_xl.max()) / 2.0
                il_half = (n_il_target // 2) * dil
                xl_half = (n_xl_target // 2) * dxl
                il_min = int(il_ctr - il_half)
                il_max = int(il_ctr + il_half)
                xl_min = int(xl_ctr - xl_half)
                xl_max = int(xl_ctr + xl_half)
                inlines = np.arange(il_min, il_max + 1, dil, dtype=np.int32)
                xlines = np.arange(xl_min, xl_max + 1, dxl, dtype=np.int32)
                new_gb = len(inlines) * len(xlines) * self._n_samp * 4 / (1024**3)
                logger.warning(
                    "Grid {:.2f} GB exceeds limit {:.2f} GB → shrunk to {} × {} ({:.2f} GB)",
                    vol_gb, max_volume_gb, len(inlines), len(xlines), new_gb,
                )

        logger.info(
            "Target grid: IL [{}, {}] step {}, XL [{}, {}] step {}, {} × {} × {}",
            il_min, il_max, dil, xl_min, xl_max, dxl,
            len(inlines), len(xlines), self._n_samp,
        )

        return TargetGrid(
            inlines=inlines,
            xlines=xlines,
            n_samples=self._n_samp,
            dt_ms=self._dt_ms,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
            il_step_x=self._il_step_x,
            il_step_y=self._il_step_y,
            xl_step_x=self._xl_step_x,
            xl_step_y=self._xl_step_y,
        )

    def inject_lines(
        self,
        grid: TargetGrid,
        lines: list[Line2D],
    ) -> SparseVolume:
        """Place calibrated 2D traces into the sparse volume at their nearest grid positions."""
        import time as _time
        t0 = _time.perf_counter()

        sparse = SparseVolume.empty(grid)
        il_arr = grid.inlines
        xl_arr = grid.xlines
        n_samp = grid.n_samples
        total_inserted = 0

        for li, line in enumerate(lines):
            il_frac, xl_frac = self._geom.xy_to_ilxl(line.coords[:, 0], line.coords[:, 1])
            # Vectorised nearest-grid lookup
            il_nearest = np.rint(il_frac).astype(np.int32)
            xl_nearest = np.rint(xl_frac).astype(np.int32)
            ii_idx = np.searchsorted(il_arr, il_nearest)
            xi_idx = np.searchsorted(xl_arr, xl_nearest)
            # Mask valid grid positions
            valid = (
                (ii_idx < len(il_arr)) & (xi_idx < len(xl_arr))
                & (il_arr[np.clip(ii_idx, 0, len(il_arr) - 1)] == il_nearest)
                & (xl_arr[np.clip(xi_idx, 0, len(xl_arr) - 1)] == xl_nearest)
            )
            for t_idx in np.where(valid)[0]:
                trace = line.data[t_idx]
                if len(trace) < n_samp:
                    trace = np.pad(trace, (0, n_samp - len(trace)))
                sparse.insert_trace(int(ii_idx[t_idx]), int(xi_idx[t_idx]), trace[:n_samp])
                total_inserted += 1

            logger.debug("Line '{}': {}/{} traces injected", line.name, int(valid.sum()), line.n_traces)

        elapsed = _time.perf_counter() - t0
        logger.info(
            "Sparse volume: {} traces injected, fill={:.2%}, elapsed={:.2f}s",
            total_inserted, sparse.fill_ratio, elapsed,
        )
        return sparse

    def inject_original_3d(
        self,
        grid: TargetGrid,
        volume: np.ndarray,
        orig_inlines: np.ndarray,
        orig_xlines: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place original 3D data into the extended grid (vectorised).

        Returns
        -------
        full_vol : (grid.n_il, grid.n_xl, grid.n_samples) with orig data placed.
        orig_mask : (grid.n_il, grid.n_xl) bool — True where original data exists.
        """
        import time as _time
        t0 = _time.perf_counter()

        full_vol = np.zeros(grid.shape, dtype=np.float32)
        orig_mask = np.zeros((grid.n_il, grid.n_xl), dtype=bool)

        # Vectorised index mapping: find where orig inlines/xlines sit in the grid
        ii_idx = np.searchsorted(grid.inlines, orig_inlines)
        xi_idx = np.searchsorted(grid.xlines, orig_xlines)

        # Validity masks
        il_valid = (ii_idx < len(grid.inlines)) & (grid.inlines[np.clip(ii_idx, 0, len(grid.inlines) - 1)] == orig_inlines)
        xl_valid = (xi_idx < len(grid.xlines)) & (grid.xlines[np.clip(xi_idx, 0, len(grid.xlines) - 1)] == orig_xlines)

        valid_il = np.where(il_valid)[0]
        valid_xl = np.where(xl_valid)[0]

        n = min(volume.shape[2], grid.n_samples)

        # Vectorised block copy — slice entire valid inlines at once
        for i in valid_il:
            gi = ii_idx[i]
            full_vol[gi, xi_idx[valid_xl], :n] = volume[i, valid_xl, :n]
            orig_mask[gi, xi_idx[valid_xl]] = True

        elapsed = _time.perf_counter() - t0
        logger.info(
            "Original 3D injected: {} / {} positions in {:.2f}s",
            orig_mask.sum(), orig_mask.size, elapsed,
        )
        return full_vol, orig_mask

    @staticmethod
    def assemble(
        orig_vol: np.ndarray,
        orig_mask: np.ndarray,
        recon_vol: np.ndarray,
        taper_width: int = 10,
        blend: bool = True,
    ) -> np.ndarray:
        """Merge original 3D and reconstructed extension.

        Inside original coverage → original data.
        Outside → reconstructed data.
        At the boundary → cosine-tapered blend (if enabled).
        """
        if blend and taper_width > 0:
            return blend_boundary(orig_vol, recon_vol, orig_mask, taper_width)

        # Hard stitch
        result = recon_vol.copy()
        result[orig_mask] = orig_vol[orig_mask]
        return result
