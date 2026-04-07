"""Tests for geometry model and overlap detection."""

import numpy as np
import pytest

from seis2cube.geometry.geometry_model import AffineGridMapper, KDTreeMapper
from seis2cube.geometry.overlap_detector import OverlapDetector
from shapely.geometry import Polygon


class TestAffineGridMapper:
    def test_roundtrip(self):
        n_il, n_xl = 5, 8
        inlines = np.arange(100, 100 + n_il)
        xlines = np.arange(200, 200 + n_xl)
        coords = np.array([
            [xl * 25.0, il * 25.0]
            for il in inlines
            for xl in xlines
        ])
        mapper = AffineGridMapper(coords, inlines, xlines)

        # Forward: (il, xl) → (x, y)
        il_q = np.array([100.0, 102.0])
        xl_q = np.array([200.0, 205.0])
        x, y = mapper.ilxl_to_xy(il_q, xl_q)

        # Inverse: (x, y) → (il, xl)
        il_r, xl_r = mapper.xy_to_ilxl(x, y)
        np.testing.assert_allclose(il_r, il_q, atol=0.5)
        np.testing.assert_allclose(xl_r, xl_q, atol=0.5)

    def test_nearest_traces(self):
        n_il, n_xl = 4, 6
        inlines = np.arange(n_il)
        xlines = np.arange(n_xl)
        coords = np.array([[xl * 10.0, il * 10.0] for il in inlines for xl in xlines])
        mapper = AffineGridMapper(coords, inlines, xlines)

        dists, idxs = mapper.nearest_traces(np.array([5.0]), np.array([5.0]), k=1)
        assert dists.shape[1] == 1
        assert idxs.shape[1] == 1


class TestKDTreeMapper:
    def test_xy_to_ilxl_nearest(self):
        coords = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
        il = np.array([0, 0, 1, 1])
        xl = np.array([0, 1, 0, 1])
        mapper = KDTreeMapper(coords, il, xl)

        il_r, xl_r = mapper.xy_to_ilxl(np.array([1.0]), np.array([1.0]))
        assert il_r[0] == 0
        assert xl_r[0] == 0


class TestOverlapDetector:
    def test_classify(self):
        cube_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        expand_poly = Polygon([(-50, -50), (200, -50), (200, 200), (-50, 200)])
        det = OverlapDetector(cube_poly, expand_poly)

        coords = np.array([
            [50, 50],     # inside 3D
            [150, 150],   # inside expand, outside 3D
            [300, 300],   # outside both
        ])
        inside, expand, outside = det.classify_line(coords)
        assert inside[0] and not inside[1] and not inside[2]
        assert expand[1] and not expand[0]
