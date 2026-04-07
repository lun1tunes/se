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
        self._dil = grid_il_step or float(self._orig_il[1] - self._orig_il[0]) if len(self._orig_il) > 1 else 1.0
        self._dxl = grid_xl_step or float(self._orig_xl[1] - self._orig_xl[0]) if len(self._orig_xl) > 1 else 1.0

        # Affine params (passed through from original 3D geometry estimation)
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._il_step_x = il_step_x
        self._il_step_y = il_step_y
        self._xl_step_x = xl_step_x
        self._xl_step_y = xl_step_y

    def build_target_grid(self) -> TargetGrid:
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
        sparse = SparseVolume.empty(grid)

        for line in lines:
            il_frac, xl_frac = self._geom.xy_to_ilxl(line.coords[:, 0], line.coords[:, 1])
            for t_idx in range(line.n_traces):
                il_nearest = int(round(il_frac[t_idx]))
                xl_nearest = int(round(xl_frac[t_idx]))
                try:
                    ii = grid.il_index(il_nearest)
                    xi = grid.xl_index(xl_nearest)
                except KeyError:
                    continue
                trace = line.data[t_idx]
                # Pad or trim to grid n_samples
                if len(trace) < grid.n_samples:
                    trace = np.pad(trace, (0, grid.n_samples - len(trace)))
                sparse.insert_trace(ii, xi, trace[:grid.n_samples])

        logger.info("Sparse volume fill ratio: {:.2%}", sparse.fill_ratio)
        return sparse

    def inject_original_3d(
        self,
        grid: TargetGrid,
        volume: np.ndarray,
        orig_inlines: np.ndarray,
        orig_xlines: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place original 3D data into the extended grid.

        Returns
        -------
        full_vol : (grid.n_il, grid.n_xl, grid.n_samples) with orig data placed.
        orig_mask : (grid.n_il, grid.n_xl) bool — True where original data exists.
        """
        full_vol = np.zeros(grid.shape, dtype=np.float32)
        orig_mask = np.zeros((grid.n_il, grid.n_xl), dtype=bool)

        for i, il in enumerate(orig_inlines):
            for j, xl in enumerate(orig_xlines):
                try:
                    ii = grid.il_index(int(il))
                    xi = grid.xl_index(int(xl))
                except KeyError:
                    continue
                n = min(volume.shape[2], grid.n_samples)
                full_vol[ii, xi, :n] = volume[i, j, :n]
                orig_mask[ii, xi] = True

        logger.info(
            "Original 3D injected: {} / {} positions",
            orig_mask.sum(), orig_mask.size,
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
