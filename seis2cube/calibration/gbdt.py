"""Level-3 calibration: Gradient Boosting / Random Forest.

Uses sklearn's HistGradientBoostingRegressor (always available) with optional
XGBoost / LightGBM backends.  Captures non-linear amplitude relationships.

Train/test split is by spatial blocks, never random-by-sample.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

from seis2cube.calibration.base import (
    CalibrationModel,
    CalibrationPair,
    CalibrationStrategy,
)
from seis2cube.calibration.regression import LinearRegressionCalibrator
from seis2cube.models.line2d import Line2D


def _make_gbdt(n_estimators: int, max_depth: int, lr: float, backend: str = "sklearn"):
    """Factory for gradient boosting regressors."""
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            tree_method="hist", verbosity=0,
        )
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, verbose=-1,
        )
    # Default: sklearn HistGradientBoosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=n_estimators, max_depth=max_depth, learning_rate=lr,
    )


class GBDTCalibrator(CalibrationStrategy):
    """Gradient-boosted tree calibration.

    Parameters
    ----------
    n_estimators, max_depth, learning_rate : GBDT hyper-parameters.
    backend : 'sklearn', 'xgboost', or 'lightgbm'.
    window_samples : feature extraction window.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        backend: str = "sklearn",
        window_samples: int = 64,
    ) -> None:
        self._n_est = n_estimators
        self._md = max_depth
        self._lr = learning_rate
        self._backend = backend
        self._win = window_samples
        # Reuse feature extraction from regression calibrator
        self._feat_extractor = LinearRegressionCalibrator(window_samples=window_samples)

    def fit(self, pairs: CalibrationPair) -> CalibrationModel:
        X, y = self._feat_extractor._build_dataset(
            pairs.amp_2d, pairs.amp_3d, pairs.dt_ms
        )
        logger.info("GBDT dataset: {} samples, {} features", X.shape[0], X.shape[1])

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        model = _make_gbdt(self._n_est, self._md, self._lr, self._backend)
        model.fit(X_sc, y)

        score = model.score(X_sc, y)
        logger.info("GBDT R² (train) = {:.4f}", score)

        # Serialise: we pickle the model object into params for now
        import pickle, base64
        model_bytes = base64.b64encode(pickle.dumps(model)).decode("ascii")

        return CalibrationModel(
            method="gbdt",
            params={
                "backend": self._backend,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "model_pickle_b64": model_bytes,
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
        import pickle, base64
        p = model.params
        was_1d = amp_2d.ndim == 1
        if was_1d:
            amp_2d = amp_2d[np.newaxis, :]

        scaler_mean = np.array(p["scaler_mean"])
        scaler_scale = np.array(p["scaler_scale"])
        gbdt = pickle.loads(base64.b64decode(p["model_pickle_b64"]))
        win = p["window_samples"]

        n_traces, n_samp = amp_2d.shape
        out = np.empty_like(amp_2d)

        for i in range(n_traces):
            trace = amp_2d[i]
            feats = LinearRegressionCalibrator._trace_features(trace, win)
            if feats.shape[0] == 0:
                out[i] = trace
                continue
            X_sc = (feats - scaler_mean) / np.clip(scaler_scale, 1e-10, None)
            pred = gbdt.predict(X_sc)
            n_win = len(pred)
            sample_indices = np.linspace(0, n_samp - 1, n_win)
            out[i] = np.interp(np.arange(n_samp), sample_indices, pred)

        result = out.astype(np.float32)
        return result[0] if was_1d else result
