"""Tests for SEG-Y I/O layer."""

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_3d_segy
from seis2cube.io.segy_dataset import SegyDataset
from seis2cube.io.segy_writer import SegyWriter3D
from seis2cube.config import SegyHeaderBytes, IOConfig


class TestSegyDataset:
    def test_open_structured(self, tmp_path: Path):
        segy_path = tmp_path / "cube.segy"
        volume = make_3d_segy(segy_path, n_il=5, n_xl=8, n_samp=50)

        with SegyDataset(segy_path) as ds:
            assert ds.n_traces == 5 * 8
            assert ds.n_samples == 50
            assert ds.meta.is_structured
            assert len(ds.meta.inlines) == 5
            assert len(ds.meta.xlines) == 8
            assert ds.meta.sample_format == 5
            assert ds.meta.sample_interval_us == 2000

    def test_coordinates(self, tmp_path: Path):
        segy_path = tmp_path / "cube.segy"
        make_3d_segy(segy_path, n_il=3, n_xl=4, n_samp=20)

        with SegyDataset(segy_path) as ds:
            coords = ds.all_coordinates()
            assert coords.shape == (12, 2)
            # First trace should be at (xl_start*25, il_start*25) = (200*25, 100*25)
            assert coords[0, 0] == pytest.approx(200 * 25.0)
            assert coords[0, 1] == pytest.approx(100 * 25.0)

    def test_read_trace(self, tmp_path: Path):
        segy_path = tmp_path / "cube.segy"
        volume = make_3d_segy(segy_path, n_il=3, n_xl=4, n_samp=30)

        with SegyDataset(segy_path) as ds:
            trace = ds.read_trace(0)
            assert len(trace) == 30
            np.testing.assert_allclose(trace, volume[0, 0], atol=1e-5)

    def test_ignore_geometry(self, tmp_path: Path):
        segy_path = tmp_path / "cube.segy"
        make_3d_segy(segy_path, n_il=3, n_xl=4, n_samp=20)
        io = IOConfig(ignore_geometry=True, strict=False)

        with SegyDataset(segy_path, io_cfg=io) as ds:
            assert not ds.meta.is_structured
            assert ds.n_traces == 12


class TestSegyWriter:
    def test_roundtrip(self, tmp_path: Path):
        volume = np.random.default_rng(0).standard_normal((4, 6, 50)).astype(np.float32)
        out_path = tmp_path / "out.segy"
        inlines = np.arange(10, 14)
        xlines = np.arange(20, 26)

        writer = SegyWriter3D(
            path=out_path,
            inlines=inlines,
            xlines=xlines,
            dt_us=4000,
        )
        writer.write(volume)

        with SegyDataset(out_path) as ds:
            assert ds.n_traces == 24
            assert ds.n_samples == 50
            assert ds.meta.sample_interval_us == 4000
            trace = ds.read_trace(0)
            np.testing.assert_allclose(trace, volume[0, 0], atol=1e-5)
