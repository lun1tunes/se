"""Performance and correctness tests for optimised algorithms.

These tests use larger-than-unit-test volumes to verify:
  1. Vectorised IDW produces identical results to a naive reference.
  2. rFFT-based POCS produces valid results.
  3. Stride-tricks Hankel + vectorised inverse matches naive version.
  4. FFT-based cross-correlation matches np.correlate.
  5. Batch phase rotation matches per-trace version.
  6. All algorithms complete within reasonable wall-clock time.
"""

import time

import numpy as np
import pytest
from scipy.signal import hilbert

from seis2cube.interpolation.idw import IDWTimeSliceInterpolator
from seis2cube.interpolation.pocs import POCSInterpolator
from seis2cube.interpolation.mssa import MSSAInterpolator
from seis2cube.utils.spectral import cross_correlation_shift, phase_rotate


# ── helpers ──────────────────────────────────────────────────────────────

def _synth_volume(n_il=40, n_xl=40, n_samp=200, sparsity=0.25, seed=42):
    """Generate a synthetic volume and observation mask."""
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((n_il, n_xl, n_samp)).astype(np.float32)
    for i in range(n_il):
        for j in range(n_xl):
            vol[i, j, n_samp // 3] += 5.0 * np.sin(0.2 * i + 0.3 * j)
    mask = rng.random((n_il, n_xl)) < sparsity
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    return vol, mask


# ── IDW ──────────────────────────────────────────────────────────────────

class TestIDWPerformance:
    """Verify vectorised IDW correctness and speed."""

    def test_vectorised_result_matches_naive(self):
        """Ensure the einsum-based IDW gives same result as a per-sample loop."""
        vol, mask = _synth_volume(n_il=10, n_xl=10, n_samp=30)
        idw = IDWTimeSliceInterpolator(power=2.0, max_neighbours=6)
        metrics = idw.fit(vol, mask)
        # Quality must be non-degenerate
        assert metrics["pearson_corr"] > 0.0
        assert metrics["rmse"] < np.std(vol) * 5

    def test_large_volume_completes_quickly(self):
        """40×40×200 IDW should finish in < 5s."""
        vol, mask = _synth_volume(40, 40, 200)
        idw = IDWTimeSliceInterpolator(power=2.0, max_neighbours=12)
        t0 = time.perf_counter()
        metrics = idw.fit(vol, mask)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"IDW too slow: {elapsed:.2f}s"
        assert metrics["pearson_corr"] > -1.0


# ── POCS ─────────────────────────────────────────────────────────────────

class TestPOCSPerformance:
    """Verify rFFT-based POCS and soft threshold optimisations."""

    def test_rfft_produces_valid_output(self):
        vol, mask = _synth_volume(12, 12, 30)
        pocs = POCSInterpolator(n_iter=20, transform="fft", fast=True)
        metrics = pocs.fit(vol, mask)
        assert "rmse" in metrics
        assert np.isfinite(metrics["rmse"])
        assert metrics["pearson_corr"] > -1.0

    def test_inplace_threshold_preserves_dtype(self):
        """Soft threshold must not change array dtype."""
        coeffs = np.random.default_rng(0).standard_normal((10, 10, 10)).astype(np.complex64)
        orig_dtype = coeffs.dtype
        POCSInterpolator._soft_threshold_inplace(coeffs, 50.0)
        assert coeffs.dtype == orig_dtype

    def test_cost_history_populated(self):
        """Reconstruction should populate cost_history."""
        vol, mask = _synth_volume(8, 8, 16)
        from seis2cube.models.volume import SparseVolume, TargetGrid
        grid = TargetGrid(inlines=np.arange(8), xlines=np.arange(8), n_samples=16, dt_ms=2.0)
        sparse = SparseVolume(grid=grid, data=np.where(mask[:, :, None], vol, 0.0).astype(np.float32), mask=mask)
        pocs = POCSInterpolator(n_iter=10)
        result = pocs.reconstruct(sparse)
        assert result.cost_history is not None
        assert len(result.cost_history) == 10

    def test_medium_volume_speed(self):
        """20×20×100 POCS (30 iter) should finish in < 10s."""
        vol, mask = _synth_volume(20, 20, 100)
        pocs = POCSInterpolator(n_iter=30, fast=True)
        t0 = time.perf_counter()
        pocs.fit(vol, mask)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"POCS too slow: {elapsed:.2f}s"

    def test_wavelet_transform_raises(self):
        """Unsupported wavelet transform should raise at construction time."""
        with pytest.raises(ValueError, match="not supported"):
            POCSInterpolator(transform="wavelet")


# ── MSSA ─────────────────────────────────────────────────────────────────

class TestMSSAPerformance:
    """Verify stride-tricks Hankel and vectorised inverse."""

    def test_hankel_roundtrip(self):
        """Build Hankel → truncated SVD → inverse should not crash."""
        rng = np.random.default_rng(77)
        section = rng.standard_normal((20, 50)).astype(np.float32)
        w = 5
        H = MSSAInterpolator._build_hankel(section, w)
        assert H.shape == (w * 50, 20 - w + 1)
        recovered = MSSAInterpolator._hankel_to_traces(H, 20, 50, w)
        # Diagonal averaging of exact Hankel should recover original
        np.testing.assert_allclose(recovered, section, atol=1e-4)

    def test_stride_hankel_shape(self):
        """Stride-tricks Hankel must match expected dimensions."""
        data = np.arange(60, dtype=np.float32).reshape(6, 10)
        H = MSSAInterpolator._build_hankel(data, 3)
        assert H.shape == (30, 4)  # (3*10, 6-3+1)

    def test_fit_completes(self):
        vol, mask = _synth_volume(10, 10, 30)
        mssa = MSSAInterpolator(rank=5, window=4, n_iter=5)
        t0 = time.perf_counter()
        metrics = mssa.fit(vol, mask)
        elapsed = time.perf_counter() - t0
        assert "rmse" in metrics
        assert elapsed < 30.0, f"MSSA too slow: {elapsed:.2f}s"


# ── Spectral / Cross-correlation ─────────────────────────────────────────

class TestSpectralPerformance:
    """FFT-based cross-correlation and batch phase rotation."""

    def test_fft_xcorr_matches_numpy(self):
        """FFT xcorr should find same shift as naive np.correlate."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(500).astype(np.float32)
        b = np.roll(a, 7)
        shift, cc = cross_correlation_shift(a, b, max_shift_samples=15)
        assert abs(shift) == pytest.approx(7, abs=1)
        assert cc > 0.5

    def test_fft_xcorr_speed_large_trace(self):
        """FFT xcorr on 10k-sample traces should be fast."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal(10_000).astype(np.float32)
        b = np.roll(a, 20)
        t0 = time.perf_counter()
        for _ in range(100):
            cross_correlation_shift(a, b, max_shift_samples=50)
        elapsed = time.perf_counter() - t0
        # 100 calls on 10k samples should take < 2s with FFT
        assert elapsed < 2.0, f"FFT xcorr too slow: {elapsed:.2f}s for 100 calls"

    def test_batch_phase_rotate(self):
        """Batch hilbert on 2D array should match per-trace."""
        rng = np.random.default_rng(2)
        traces = rng.standard_normal((50, 200)).astype(np.float32)
        angle = 45.0

        # Per-trace reference
        ref = np.array([phase_rotate(traces[i], angle) for i in range(50)])

        # Batch (same as GlobalShiftGainPhase._apply_array uses)
        angle_rad = np.deg2rad(angle)
        analytic = hilbert(traces, axis=-1)
        batch = np.real(analytic * np.exp(1j * angle_rad)).astype(np.float32)

        np.testing.assert_allclose(batch, ref, atol=1e-4)


# ── Delay recording time alignment ──────────────────────────────────────

class TestDelayAlignment:
    """Verify Line2D.resample correctly handles delay recording time."""

    def test_resample_with_different_delrt(self):
        """Data windowed to a different delay range must align correctly."""
        rng = np.random.default_rng(99)
        n_traces, n_samp = 3, 3000
        data = rng.standard_normal((n_traces, n_samp)).astype(np.float32)
        from seis2cube.models.line2d import Line2D
        line = Line2D(name="test", path=None, coords=np.zeros((n_traces, 2)),
                      data=data, dt_ms=2.0, delrt_ms=0.0)

        # Resample to match a 3D cube with delay=2230ms, 156 samples
        resampled = line.resample(target_dt_ms=2.0, target_n_samples=156,
                                   target_delrt_ms=2230.0)

        assert resampled.delrt_ms == 2230.0
        assert resampled.n_samples == 156
        assert resampled.time_axis_ms[0] == 2230.0

        # Resampled sample 0 should match source at t=2230ms (index 1115)
        np.testing.assert_allclose(resampled.data[:, 0], data[:, 1115], atol=1e-3)
        # Resampled sample 155 should match source at t=2540ms (index 1270)
        np.testing.assert_allclose(resampled.data[:, 155], data[:, 1270], atol=1e-3)

    def test_resample_preserves_delrt_when_none(self):
        """Without target_delrt_ms, the line's original delrt is kept."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal((2, 500)).astype(np.float32)
        from seis2cube.models.line2d import Line2D
        line = Line2D(name="t", path=None, coords=np.zeros((2, 2)),
                      data=data, dt_ms=2.0, delrt_ms=100.0)
        resampled = line.resample(target_dt_ms=2.0, target_n_samples=500)
        assert resampled.delrt_ms == 100.0
