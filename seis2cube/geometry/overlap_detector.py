"""Detect overlap between 2D line coordinates and 3D cube coverage polygon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon, shape
from shapely.ops import split

from loguru import logger


class OverlapDetector:
    """Determines which segments of 2D lines fall inside the 3D coverage area.

    The 3D coverage is represented as a Shapely Polygon (convex hull of 3D coords
    or user-provided boundary).  An optional ``expand_polygon`` defines the full
    target area (3D + extension).
    """

    def __init__(
        self,
        cube3d_polygon: Polygon,
        expand_polygon: Polygon | None = None,
    ) -> None:
        self._cube_poly = cube3d_polygon
        self._expand_poly = expand_polygon

    # -- factories -----------------------------------------------------------

    @classmethod
    def from_3d_coords(
        cls,
        coords: np.ndarray,
        expand_polygon: Polygon | None = None,
        buffer: float = 0.0,
    ) -> "OverlapDetector":
        """Build from the (N, 2) coordinate array of the 3D cube's traces."""
        hull = MultiPoint(coords).convex_hull
        if buffer > 0:
            hull = hull.buffer(buffer)
        if not isinstance(hull, Polygon):
            hull = hull.convex_hull
        return cls(cube3d_polygon=hull, expand_polygon=expand_polygon)

    @classmethod
    def load_polygon(cls, path: str | Path) -> Polygon:
        """Load a polygon from GeoJSON, WKT text file, or Shapefile."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".geojson" or suffix == ".json":
            import json
            with open(path) as f:
                geoj = json.load(f)
            # Handle FeatureCollection or single Feature or bare geometry
            if geoj.get("type") == "FeatureCollection":
                geom = geoj["features"][0]["geometry"]
            elif geoj.get("type") == "Feature":
                geom = geoj["geometry"]
            else:
                geom = geoj
            return shape(geom)

        if suffix == ".wkt":
            from shapely import wkt
            with open(path) as f:
                return wkt.loads(f.read())

        if suffix in (".shp", ".gpkg"):
            import geopandas as gpd
            gdf = gpd.read_file(path)
            return gdf.geometry.iloc[0]

        raise ValueError(f"Unsupported polygon file format: {suffix}")

    # -- overlap queries -----------------------------------------------------

    def classify_line(
        self, coords_2d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify each point on a 2D line as inside/outside the 3D coverage.

        Parameters
        ----------
        coords_2d : (M, 2) array of (X, Y) along the 2D profile.

        Returns
        -------
        inside_mask : (M,) bool — True where point is inside 3D coverage.
        expand_mask : (M,) bool — True where point is inside expansion polygon
                      but outside 3D.  (All False if no expand_polygon.)
        outside_mask : (M,) bool — True where outside both.
        """
        from shapely import contains_xy

        inside = contains_xy(self._cube_poly, coords_2d[:, 0], coords_2d[:, 1])

        if self._expand_poly is not None:
            in_expand = contains_xy(self._expand_poly, coords_2d[:, 0], coords_2d[:, 1])
            expand_mask = in_expand & ~inside
        else:
            expand_mask = np.zeros(len(coords_2d), dtype=bool)

        outside = ~inside & ~expand_mask
        return inside, expand_mask, outside

    def overlap_indices(self, coords_2d: np.ndarray) -> np.ndarray:
        """Return indices of 2D trace points that fall inside the 3D area."""
        inside, _, _ = self.classify_line(coords_2d)
        return np.nonzero(inside)[0]

    def expansion_indices(self, coords_2d: np.ndarray) -> np.ndarray:
        """Return indices within the expansion polygon but outside 3D."""
        _, expand, _ = self.classify_line(coords_2d)
        return np.nonzero(expand)[0]

    @property
    def cube_polygon(self) -> Polygon:
        return self._cube_poly

    @property
    def expand_polygon(self) -> Polygon | None:
        return self._expand_poly
