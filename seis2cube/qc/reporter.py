"""QC reporting: save metrics, tables, and optional quick-look slices."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from loguru import logger

from seis2cube.qc.metrics import CalibrationMetrics, InterpolationMetrics


class QCReporter:
    """Collect and persist QC artifacts for the pipeline run."""

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cal_metrics: CalibrationMetrics | None = None
        self._interp_metrics: InterpolationMetrics | None = None
        self._extra: dict = {}

    def log_calibration(
        self,
        baseline_corr: float,
        baseline_rmse: float,
        after_metrics: dict[str, float],
    ) -> None:
        self._cal_metrics = CalibrationMetrics(
            baseline_corr=baseline_corr,
            baseline_rmse=baseline_rmse,
            calibrated_corr=after_metrics.get("pearson_corr", 0.0),
            calibrated_rmse=after_metrics.get("rmse", 0.0),
            calibrated_mae=after_metrics.get("mae", 0.0),
            spectral_l2_rel=after_metrics.get("spectral_l2_rel", 0.0),
        )

    def log_interpolation_sim(self, metrics: dict[str, float]) -> None:
        self._interp_metrics = InterpolationMetrics(
            rmse=metrics.get("rmse", 0.0),
            mae=metrics.get("mae", 0.0),
            pearson_corr=metrics.get("pearson_corr", 0.0),
        )

    def add_extra(self, key: str, value) -> None:
        self._extra[key] = value

    def save(self, config=None) -> None:
        """Write all collected metrics to disk."""
        report = {}

        if self._cal_metrics is not None:
            report["calibration"] = self._cal_metrics.to_dict()
        if self._interp_metrics is not None:
            report["interpolation_simulation"] = self._interp_metrics.to_dict()
        if self._extra:
            report["extra"] = self._extra
        if config is not None:
            report["config_summary"] = {
                "calibration_method": str(config.calibration.method.value),
                "interpolation_method": str(config.interpolation.method.value),
                "blend_enabled": config.blend.enabled,
            }

        # JSON report
        json_path = self._dir / "qc_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("QC report saved: {}", json_path)

        # CSV summary
        csv_path = self._dir / "qc_summary.csv"
        flat = self._flatten(report)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in flat.items():
                w.writerow([k, v])
        logger.info("QC CSV saved: {}", csv_path)

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(QCReporter._flatten(v, key))
            else:
                out[key] = v
        return out
