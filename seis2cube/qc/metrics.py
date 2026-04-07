"""Calibration and interpolation quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationMetrics:
    """Holds before/after calibration comparison numbers."""
    baseline_corr: float = 0.0
    baseline_rmse: float = 0.0
    calibrated_corr: float = 0.0
    calibrated_rmse: float = 0.0
    calibrated_mae: float = 0.0
    spectral_l2_rel: float = 0.0

    def improvement_corr(self) -> float:
        return self.calibrated_corr - self.baseline_corr

    def improvement_rmse(self) -> float:
        return self.baseline_rmse - self.calibrated_rmse

    def to_dict(self) -> dict[str, float]:
        return {
            "baseline_corr": self.baseline_corr,
            "baseline_rmse": self.baseline_rmse,
            "calibrated_corr": self.calibrated_corr,
            "calibrated_rmse": self.calibrated_rmse,
            "calibrated_mae": self.calibrated_mae,
            "spectral_l2_rel": self.spectral_l2_rel,
            "improvement_corr": self.improvement_corr(),
            "improvement_rmse": self.improvement_rmse(),
        }


@dataclass
class InterpolationMetrics:
    """Holds interpolation simulation quality numbers."""
    rmse: float = 0.0
    mae: float = 0.0
    pearson_corr: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "pearson_corr": self.pearson_corr,
        }
