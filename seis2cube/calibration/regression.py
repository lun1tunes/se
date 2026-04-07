"""Level-2 calibration: feature-based linear regression.

Builds a dataset of (features_2d, amp_3d) pairs per window/trace and
trains Ridge / ElasticNet / Huber regression.  Features include amplitude,
envelope, RMS, d/dt, etc.

IMPORTANT: train/test split is spatial (by trace blocks), not random by sample.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler

from seis2cube.calibration.base import (
    CalibrationModel,
    CalibrationPair,
    CalibrationStrategy,
)
from seis2cube.models.line2d import Line2D
from seis2cube.utils.spectral import envelope, rms_amplitude


_REGRESSORS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "elasticnet": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
    "huber": lambda: HuberRegressor(max_iter=200),
}


class LinearRegressionCalibrator(CalibrationStrategy):
    """Regression-based amplitude calibration.

    Parameters
    ----------
    regressor_name : one of 'ridge', 'elasticnet', 'huber'.
    window_samples : feature extraction window (samples).
    """

    def __init__(
        self,
        regressor_name: str = "ridge",
        window_samples: int = 64,
    ) -> None:
        self._reg_name = regressor_name
        self._win = window_samples

    def fit(self, pairs: CalibrationPair) -> CalibrationModel:
        X, y = self._build_dataset(pairs.amp_2d, pairs.amp_3d, pairs.dt_ms)
        logger.info("Regression dataset: {} samples, {} features", X.shape[0], X.shape[1])

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        reg = _REGRESSORS[self._reg_name]()
        reg.fit(X_sc, y)

        score = reg.score(X_sc, y)
        logger.info("Regression R² (train) = {:.4f}", score)

        return CalibrationModel(
            method="linear_regression",
            params={
                "regressor_name": self._reg_name,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coef": reg.coef_.tolist() if hasattr(reg, "coef_") else [],
                "intercept": float(reg.intercept_),
                "window_samples": self._win,
            },
        )

    def apply(self, line: Line2D, model: CalibrationModel) -> Line2D:
        corrected = self._apply_array(line.data, model)
        return Line2D(
            name=line.name,
            path=line.path,
            coords=line.coords.copy(),
            data=corrected,
            dt_ms=line.dt_ms,
            delrt_ms=line.delrt_ms,
            quality_flags=line.quality_flags,
        )

    def _apply_array(self, amp_2d: np.ndarray, model: CalibrationModel) -> np.ndarray:
        p = model.params
        scaler_mean = np.array(p["scaler_mean"])
        scaler_scale = np.array(p["scaler_scale"])
        coef = np.array(p["coef"])
        intercept = p["intercept"]
        win = p["window_samples"]

        n_traces, n_samp = amp_2d.shape
        out = np.empty_like(amp_2d)

        for i in range(n_traces):
            trace = amp_2d[i]
            feats = self._trace_features(trace, win)  # (n_windows, n_feat)
            X_sc = (feats - scaler_mean) / np.clip(scaler_scale, 1e-10, None)
            pred = X_sc @ coef + intercept  # (n_windows,)
            # Map window predictions back to samples (nearest window center)
            n_win = len(pred)
            sample_indices = np.linspace(0, n_samp - 1, n_win)
            out[i] = np.interp(np.arange(n_samp), sample_indices, pred)

        return out.astype(np.float32)

    # -- internals -----------------------------------------------------------

    def _build_dataset(
        self, amp_2d: np.ndarray, amp_3d: np.ndarray, dt_ms: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) feature matrix from matched pairs."""
        all_X, all_y = [], []
        n_traces, n_samp = amp_2d.shape
        for i in range(n_traces):
            feats = self._trace_features(amp_2d[i], self._win)
            targets = self._trace_targets(amp_3d[i], self._win)
            all_X.append(feats)
            all_y.append(targets)
        return np.vstack(all_X), np.concatenate(all_y)

    @staticmethod
    def _trace_features(trace: np.ndarray, win: int) -> np.ndarray:
        """Extract feature vectors for non-overlapping windows of a trace."""
        n = len(trace)
        env = envelope(trace)
        grad = np.gradient(trace)
        feats = []
        for s in range(0, n - win + 1, win):
            w = trace[s: s + win]
            we = env[s: s + win]
            wg = grad[s: s + win]
            feats.append([
                np.mean(w),
                np.std(w),
                np.sqrt(np.mean(w ** 2)),
                np.mean(we),
                np.max(we),
                np.sqrt(np.mean(wg ** 2)),
            ])
        return np.array(feats, dtype=np.float64) if feats else np.empty((0, 6), dtype=np.float64)

    @staticmethod
    def _trace_targets(trace: np.ndarray, win: int) -> np.ndarray:
        """Mean 3D amplitude per window — the regression target."""
        n = len(trace)
        targets = []
        for s in range(0, n - win + 1, win):
            targets.append(np.mean(trace[s: s + win]))
        return np.array(targets, dtype=np.float64)
