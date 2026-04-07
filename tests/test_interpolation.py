"""Tests for interpolation strategies."""

import numpy as np
import pytest

from seis2cube.interpolation.idw import IDWTimeSliceInterpolator
from seis2cube.interpolation.pocs import POCSInterpolator
from seis2cube.interpolation.mssa import MSSAInterpolator
from seis2cube.models.volume import SparseVolume, TargetGrid


def _make_volume_and_mask(
    n_il: int = 20,
    n_xl: int = 20,
    n_samp: int = 50,
    sparsity: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 3D volume and a random observation mask."""
    rng = np.random.default_rng(77)
    vol = rng.standard_normal((n_il, n_xl, n_samp)).astype(np.float32)
    # Add smooth structure
    for i in range(n_il):
        for j in range(n_xl):
            vol[i, j, n_samp // 3] += 5.0 * np.sin(0.2 * i + 0.3 * j)

    mask = rng.random((n_il, n_xl)) < sparsity
    # Ensure at least some observations
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    return vol, mask


class TestIDW:
    def test_reconstructs_with_nonzero_corr(self):
        vol, mask = _make_volume_and_mask()
        idw = IDWTimeSliceInterpolator(power=2.0, max_neighbours=8)
        metrics = idw.fit(vol, mask)
        assert metrics["pearson_corr"] > 0.0
        assert metrics["rmse"] < np.std(vol) * 3

    def test_reconstruct_sparse_volume(self):
        vol, mask = _make_volume_and_mask(n_il=10, n_xl=10, n_samp=20)
        grid = TargetGrid(
            inlines=np.arange(10), xlines=np.arange(10),
            n_samples=20, dt_ms=2.0,
        )
        sparse = SparseVolume(grid=grid, data=vol.copy(), mask=mask)
        sparse.data[~mask] = np.nan

        idw = IDWTimeSliceInterpolator()
        result = idw.reconstruct(sparse)
        assert result.volume.shape == vol.shape
        assert np.all(np.isfinite(result.volume[mask]))


class TestPOCS:
    def test_fit_metrics(self):
        vol, mask = _make_volume_and_mask(n_il=12, n_xl=12, n_samp=30)
        pocs = POCSInterpolator(n_iter=20, transform="fft", fast=True)
        metrics = pocs.fit(vol, mask)
        assert "rmse" in metrics
        assert metrics["pearson_corr"] > -1.0

    def test_reconstruct(self):
        vol, mask = _make_volume_and_mask(n_il=8, n_xl=8, n_samp=16)
        grid = TargetGrid(
            inlines=np.arange(8), xlines=np.arange(8),
            n_samples=16, dt_ms=2.0,
        )
        sparse = SparseVolume(grid=grid, data=np.where(mask[:, :, None], vol, 0.0).astype(np.float32), mask=mask)
        pocs = POCSInterpolator(n_iter=10)
        result = pocs.reconstruct(sparse)
        assert result.volume.shape == vol.shape


class TestMSSA:
    def test_fit(self):
        vol, mask = _make_volume_and_mask(n_il=10, n_xl=10, n_samp=30)
        mssa = MSSAInterpolator(rank=5, window=4, n_iter=5)
        metrics = mssa.fit(vol, mask)
        assert "rmse" in metrics
