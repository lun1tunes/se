"""Tests for spectral / signal-processing utilities."""

import numpy as np
import pytest

from seis2cube.utils.spectral import (
    amplitude_spectrum,
    cross_correlation_shift,
    envelope,
    phase_rotate,
)


def test_envelope_nonnegative():
    trace = np.sin(np.linspace(0, 4 * np.pi, 200)).astype(np.float32)
    env = envelope(trace)
    assert np.all(env >= -1e-7)


def test_cross_correlation_shift_detects_known_shift():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(100).astype(np.float32)
    b = np.roll(a, 5)
    shift, cc = cross_correlation_shift(a, b, max_shift_samples=10)
    assert abs(shift) == pytest.approx(5, abs=1)
    assert cc > 0.5


def test_phase_rotate_identity():
    trace = np.sin(np.linspace(0, 2 * np.pi, 100)).astype(np.float32)
    rotated = phase_rotate(trace, 0.0)
    np.testing.assert_allclose(rotated, trace, atol=1e-5)


def test_amplitude_spectrum_shape():
    trace = np.random.default_rng(1).standard_normal(128).astype(np.float32)
    freqs, spec = amplitude_spectrum(trace, dt_ms=2.0)
    assert len(freqs) == len(spec)
    assert freqs[0] == 0.0
