"""Tests for calibration strategies."""

import numpy as np
import pytest

from seis2cube.calibration.base import CalibrationPair
from seis2cube.calibration.global_shift import GlobalShiftGainPhase
from seis2cube.calibration.windowed import WindowedShiftGain
from seis2cube.calibration.regression import LinearRegressionCalibrator
from seis2cube.models.line2d import Line2D


def _make_pairs(
    n_traces: int = 50,
    n_samp: int = 200,
    dt_ms: float = 2.0,
    gain: float = 1.5,
    shift: int = 3,
) -> CalibrationPair:
    """Create synthetic calibration pairs with known shift/gain."""
    rng = np.random.default_rng(99)
    ref = rng.standard_normal((n_traces, n_samp)).astype(np.float32)
    # Add a clear event
    ref[:, n_samp // 3] += 10.0
    ref[:, 2 * n_samp // 3] -= 7.0

    # 2D = shifted + scaled version of 3D
    amp_2d = np.roll(ref, shift, axis=-1) / gain + rng.normal(0, 0.05, ref.shape).astype(np.float32)

    coords = np.column_stack([
        np.arange(n_traces) * 25.0,
        np.ones(n_traces) * 100.0,
    ])
    return CalibrationPair(coords=coords, amp_2d=amp_2d, amp_3d=ref, dt_ms=dt_ms)


class TestGlobalShift:
    def test_recovers_shift_and_gain(self):
        pairs = _make_pairs(gain=2.0, shift=5)
        cal = GlobalShiftGainPhase(max_shift_ms=20.0, estimate_phase=False)
        model = cal.fit(pairs)

        assert model.params["shift_samples"] == pytest.approx(-5, abs=1)
        assert model.params["gain"] == pytest.approx(2.0, rel=0.3)

    def test_apply_improves_correlation(self):
        pairs = _make_pairs(gain=1.5, shift=3)
        cal = GlobalShiftGainPhase(estimate_phase=False)
        model = cal.fit(pairs)

        metrics = cal.evaluate(pairs, model)
        assert metrics["pearson_corr"] > 0.5


class TestWindowed:
    def test_fit_produces_windows(self):
        pairs = _make_pairs()
        cal = WindowedShiftGain(window_ms=100.0, overlap_ms=20.0)
        model = cal.fit(pairs)
        assert len(model.params["window_centers_ms"]) > 0

    def test_apply_runs(self):
        pairs = _make_pairs()
        cal = WindowedShiftGain()
        model = cal.fit(pairs)
        line = Line2D(
            name="test", path=None,
            coords=pairs.coords, data=pairs.amp_2d,
            dt_ms=pairs.dt_ms,
        )
        corrected = cal.apply(line, model)
        assert corrected.data.shape == pairs.amp_2d.shape


class TestRegression:
    def test_fit_and_evaluate(self):
        pairs = _make_pairs(n_traces=30, n_samp=128)
        cal = LinearRegressionCalibrator(window_samples=32)
        model = cal.fit(pairs)
        metrics = cal.evaluate(pairs, model)
        assert "rmse" in metrics
        assert "pearson_corr" in metrics
