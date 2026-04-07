"""PipelineRunner: orchestrates the full seis2cube workflow.

Steps:
  1. Ingest & Validate (open 3D SEG-Y + 2D SEG-Y files).
  2. Geo-reference (CRS conversion, geometry model construction).
  3. Overlap detection (classify 2D trace points vs 3D coverage).
  4. Build calibration pairs (match 2D ↔ 3D amplitudes in overlap).
  5. Calibrate (train selected strategy, evaluate on holdout).
  6. Apply calibration to all 2D lines.
  7. Build target grid and sparse volume.
  8. Tune interpolation inside 3D (mask simulation).
  9. Reconstruct extension.
  10. Assemble & write output SEG-Y.
  11. QC report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from seis2cube.calibration.base import CalibrationPair, CalibrationStrategy
from seis2cube.calibration.gbdt import GBDTCalibrator
from seis2cube.calibration.global_shift import GlobalShiftGainPhase
from seis2cube.calibration.regression import LinearRegressionCalibrator
from seis2cube.calibration.windowed import WindowedShiftGain
from seis2cube.config import CalibrationMethod, InterpolationMethod, PipelineConfig
from seis2cube.geometry.crs_converter import CRSConverter
from seis2cube.geometry.geometry_model import AffineGridMapper, KDTreeMapper
from seis2cube.geometry.overlap_detector import OverlapDetector
from seis2cube.interpolation.base import InterpolationStrategy
from seis2cube.interpolation.idw import IDWTimeSliceInterpolator
from seis2cube.interpolation.mssa import MSSAInterpolator
from seis2cube.interpolation.pocs import POCSInterpolator
from seis2cube.io.segy_dataset import SegyDataset
from seis2cube.io.segy_writer import SegyWriter3D
from seis2cube.models.line2d import Line2D
from seis2cube.pipeline.volume_builder import VolumeBuilder
from seis2cube.qc.reporter import QCReporter


def _make_calibrator(cfg: PipelineConfig) -> CalibrationStrategy:
    """Factory for calibration strategies based on config."""
    cc = cfg.calibration
    if cc.method == CalibrationMethod.GLOBAL_SHIFT:
        return GlobalShiftGainPhase(
            max_shift_ms=50.0,
            estimate_phase=True,
            estimate_matching_filter=cc.spectral_matching,
        )
    if cc.method == CalibrationMethod.WINDOWED:
        return WindowedShiftGain(
            window_ms=cc.window_ms,
            overlap_ms=cc.window_overlap_ms,
        )
    if cc.method == CalibrationMethod.LINEAR_REGRESSION:
        return LinearRegressionCalibrator(regressor_name="ridge")
    if cc.method == CalibrationMethod.GBDT:
        return GBDTCalibrator(
            n_estimators=cc.n_estimators,
            max_depth=cc.max_depth,
            learning_rate=cc.learning_rate,
        )
    raise ValueError(f"Unsupported calibration method: {cc.method}")


def _make_interpolator(cfg: PipelineConfig) -> InterpolationStrategy:
    """Factory for interpolation strategies based on config."""
    ic = cfg.interpolation
    if ic.method == InterpolationMethod.IDW:
        return IDWTimeSliceInterpolator(power=ic.idw_power, max_neighbours=ic.idw_max_neighbours)
    if ic.method == InterpolationMethod.POCS:
        return POCSInterpolator(
            n_iter=ic.pocs_niter,
            transform=ic.pocs_transform.value,
            fast=ic.pocs_fast,
            threshold_schedule=ic.pocs_threshold_schedule,
        )
    if ic.method == InterpolationMethod.MSSA:
        return MSSAInterpolator(rank=ic.mssa_rank, window=ic.mssa_window)
    raise ValueError(f"Unsupported interpolation method: {ic.method}")


class PipelineRunner:
    """Main entry point for the seis2cube pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._qc = QCReporter(config.qc_report_dir)

    def run(self) -> Path:
        """Execute the full pipeline and return the output SEG-Y path."""
        cfg = self._cfg

        # ── 1. Ingest 3D ─────────────────────────────────────────────────
        logger.info("=== Step 1: Ingesting 3D cube ===")
        with SegyDataset(cfg.cube3d_path, cfg.header_bytes, cfg.io) as ds3d:
            meta3d = ds3d.meta
            logger.info(
                "3D cube: {} traces, {} samples, dt={} ms, format={}",
                meta3d.n_traces, meta3d.n_samples, meta3d.dt_ms, meta3d.sample_format,
            )

            coords_3d = ds3d.all_coordinates()
            ilxl_3d = ds3d.all_inlines_xlines()

            # Load full 3D volume into memory (for calibration/simulation)
            # For very large cubes, this should use chunked loading; here
            # we use inline-by-inline loading.
            cube_volume = self._load_3d_volume(ds3d)
            inlines_3d = meta3d.inlines
            xlines_3d = meta3d.xlines

        # ── 2. CRS conversion ────────────────────────────────────────────
        logger.info("=== Step 2: CRS conversion ===")
        crs_conv = CRSConverter(cfg.crs)
        if not crs_conv.is_identity:
            cx, cy = crs_conv.forward(coords_3d[:, 0], coords_3d[:, 1])
            coords_3d = np.column_stack([cx, cy])
            logger.info("Coordinates reprojected to {}", cfg.crs.target_crs)

        # ── 3. Build geometry model ──────────────────────────────────────
        logger.info("=== Step 3: Building geometry model ===")
        if inlines_3d is not None and xlines_3d is not None:
            geom = AffineGridMapper(coords_3d, inlines_3d, xlines_3d)
            logger.info("Using AffineGridMapper")
        else:
            assert ilxl_3d is not None
            geom = KDTreeMapper(coords_3d, ilxl_3d[:, 0], ilxl_3d[:, 1])
            logger.info("Using KDTreeMapper (fallback)")

        # ── 4. Load expansion polygon & overlap detector ─────────────────
        logger.info("=== Step 4: Overlap detection setup ===")
        expand_poly = OverlapDetector.load_polygon(cfg.expand_polygon_path)
        if not crs_conv.is_identity:
            from shapely.ops import transform as shp_transform
            expand_poly = shp_transform(
                lambda x, y: crs_conv.forward(np.array(x), np.array(y)), expand_poly
            )

        overlap = OverlapDetector.from_3d_coords(coords_3d, expand_polygon=expand_poly)

        # ── 5. Ingest 2D lines ───────────────────────────────────────────
        logger.info("=== Step 5: Ingesting 2D lines ===")
        lines_2d: list[Line2D] = []
        for lpath in cfg.lines2d_paths:
            line = self._load_2d_line(lpath, crs_conv, meta3d.dt_ms, meta3d.n_samples)
            lines_2d.append(line)
            logger.info("  2D line '{}': {} traces", line.name, line.n_traces)

        # ── 6. Build calibration pairs ───────────────────────────────────
        logger.info("=== Step 6: Building calibration pairs ===")
        all_train_pairs, all_test_pairs = self._build_calibration_pairs(
            lines_2d, overlap, geom, ds3d_meta=meta3d,
            cube_volume=cube_volume, inlines_3d=inlines_3d, xlines_3d=xlines_3d,
        )

        # ── 7. Calibrate ────────────────────────────────────────────────
        logger.info("=== Step 7: Calibration ===")
        calibrator = _make_calibrator(cfg)
        cal_model = calibrator.fit(all_train_pairs)
        logger.info("Calibration model fitted: {}", cal_model.method)

        if all_test_pairs.amp_2d.shape[0] > 0:
            metrics_before = calibrator.evaluate(all_test_pairs, cal_model)
            # Baseline: evaluate without calibration
            from seis2cube.calibration.base import CalibrationModel
            identity_model = CalibrationModel(method="identity", params={
                "shift_samples": 0, "shift_ms": 0.0, "gain": 1.0, "phase_deg": 0.0,
                "matching_filter": None,
            })
            # For baseline we just compare raw 2D vs 3D
            ref = all_test_pairs.amp_3d
            raw = all_test_pairs.amp_2d
            corrs_raw = []
            for i in range(len(raw)):
                c = np.corrcoef(raw[i], ref[i])[0, 1]
                if np.isfinite(c):
                    corrs_raw.append(c)
            baseline_corr = float(np.mean(corrs_raw)) if corrs_raw else 0.0
            baseline_rmse = float(np.sqrt(np.mean((raw - ref) ** 2)))

            logger.info("Calibration test — before: corr={:.3f} RMSE={:.4f}", baseline_corr, baseline_rmse)
            logger.info("Calibration test — after:  corr={:.3f} RMSE={:.4f}",
                        metrics_before["pearson_corr"], metrics_before["rmse"])
            self._qc.log_calibration(baseline_corr, baseline_rmse, metrics_before)

        # ── 8. Apply calibration to all 2D lines ────────────────────────
        logger.info("=== Step 8: Applying calibration ===")
        calibrated_lines = [calibrator.apply(line, cal_model) for line in lines_2d]

        # ── 9. Build target grid & sparse volume ────────────────────────
        logger.info("=== Step 9: Building target grid ===")
        # Estimate affine params from 3D geometry
        n_il_orig = len(inlines_3d)
        n_xl_orig = len(xlines_3d)
        p00 = coords_3d[0]
        p0n = coords_3d[n_xl_orig - 1]
        pn0 = coords_3d[(n_il_orig - 1) * n_xl_orig]
        il_step_xy = (pn0 - p00) / max(n_il_orig - 1, 1)
        xl_step_xy = (p0n - p00) / max(n_xl_orig - 1, 1)

        vb = VolumeBuilder(
            geometry=geom,
            orig_inlines=inlines_3d,
            orig_xlines=xlines_3d,
            n_samples=meta3d.n_samples,
            dt_ms=meta3d.dt_ms,
            expand_polygon=expand_poly,
            origin_x=float(p00[0]),
            origin_y=float(p00[1]),
            il_step_x=float(il_step_xy[0]),
            il_step_y=float(il_step_xy[1]),
            xl_step_x=float(xl_step_xy[0]),
            xl_step_y=float(xl_step_xy[1]),
        )

        target_grid = vb.build_target_grid()
        sparse = vb.inject_lines(target_grid, calibrated_lines)

        # Also inject original 3D into the extended grid
        orig_in_grid, orig_mask = vb.inject_original_3d(
            target_grid, cube_volume, inlines_3d, xlines_3d,
        )

        # ── 10. Tune interpolation on 3D (mask simulation) ──────────────
        logger.info("=== Step 10: Tuning interpolation on 3D simulation ===")
        interpolator = _make_interpolator(cfg)

        # Create a simulation mask inside the original 3D area
        sim_mask = self._create_simulation_mask(
            orig_mask, sparse.mask, inlines_3d, xlines_3d, target_grid,
        )
        sim_metrics = interpolator.fit(orig_in_grid, sim_mask)
        logger.info("Interpolation simulation metrics: {}", sim_metrics)
        self._qc.log_interpolation_sim(sim_metrics)

        # ── 11. Reconstruct extension ────────────────────────────────────
        logger.info("=== Step 11: Reconstructing extension ===")
        # Combine orig + sparse into one volume for reconstruction
        combined_data = orig_in_grid.copy()
        combined_mask = orig_mask.copy()
        # Add sparse observations from 2D lines (outside orig)
        ext_only = sparse.mask & ~orig_mask
        combined_data[ext_only] = np.nan_to_num(sparse.data[ext_only], nan=0.0)
        combined_mask[ext_only] = True

        from seis2cube.models.volume import SparseVolume as SV
        combined_sparse = SV(grid=target_grid, data=combined_data, mask=combined_mask)
        result = interpolator.reconstruct(combined_sparse)

        # ── 12. Assemble final cube ──────────────────────────────────────
        logger.info("=== Step 12: Assembling final cube ===")
        final = VolumeBuilder.assemble(
            orig_vol=orig_in_grid,
            orig_mask=orig_mask,
            recon_vol=result.volume,
            taper_width=cfg.blend.taper_width_traces if cfg.blend.enabled else 0,
            blend=cfg.blend.enabled,
        )

        # ── 13. Write output SEG-Y ──────────────────────────────────────
        logger.info("=== Step 13: Writing output SEG-Y ===")
        writer = SegyWriter3D(
            path=cfg.out_cube_path,
            inlines=target_grid.inlines,
            xlines=target_grid.xlines,
            dt_us=int(meta3d.dt_ms * 1000),
            header_bytes=cfg.header_bytes,
            origin_x=target_grid.origin_x,
            origin_y=target_grid.origin_y,
            il_step_x=target_grid.il_step_x,
            il_step_y=target_grid.il_step_y,
            xl_step_x=target_grid.xl_step_x,
            xl_step_y=target_grid.xl_step_y,
        )
        out_path = writer.write(final)

        # ── 14. QC report ────────────────────────────────────────────────
        logger.info("=== Step 14: QC report ===")
        self._qc.save(cfg)

        logger.info("Pipeline complete. Output: {}", out_path)
        return out_path

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_3d_volume(ds: SegyDataset) -> np.ndarray:
        """Load a structured 3D SEG-Y into (n_il, n_xl, n_samp) array."""
        meta = ds.meta
        if not meta.is_structured:
            raise RuntimeError("3D cube must be opened in structured mode (check header bytes / strict)")
        n_il = len(meta.inlines)
        n_xl = len(meta.xlines)
        n_samp = meta.n_samples
        vol = np.empty((n_il, n_xl, n_samp), dtype=np.float32)
        for i, il in enumerate(meta.inlines):
            vol[i] = ds.read_inline(int(il))
        logger.info("3D volume loaded: shape={}", vol.shape)
        return vol

    def _load_2d_line(
        self,
        path: Path,
        crs_conv: CRSConverter,
        target_dt_ms: float,
        target_n_samples: int,
    ) -> Line2D:
        """Load a single 2D SEG-Y profile."""
        from seis2cube.config import IOConfig, SegyHeaderBytes
        io_cfg = IOConfig(
            mmap=self._cfg.io.mmap,
            ignore_geometry=True,  # 2D lines: no inline/xline geometry
            strict=False,
        )
        with SegyDataset(path, self._cfg.header_bytes, io_cfg) as ds:
            coords = ds.all_coordinates()
            if not crs_conv.is_identity:
                cx, cy = crs_conv.forward(coords[:, 0], coords[:, 1])
                coords = np.column_stack([cx, cy])

            n = ds.n_traces
            data = np.empty((n, ds.n_samples), dtype=np.float32)
            for i in range(n):
                data[i] = ds.read_trace(i)

            line = Line2D(
                name=path.stem,
                path=path,
                coords=coords,
                data=data,
                dt_ms=ds.dt_ms,
            )

        # Resample to match 3D if needed
        if abs(line.dt_ms - target_dt_ms) > 0.01 or line.n_samples != target_n_samples:
            logger.info(
                "Resampling '{}': dt {:.2f}→{:.2f} ms, nsamp {}→{}",
                line.name, line.dt_ms, target_dt_ms, line.n_samples, target_n_samples,
            )
            line = line.resample(target_dt_ms, target_n_samples)

        return line

    def _build_calibration_pairs(
        self,
        lines: list[Line2D],
        overlap: OverlapDetector,
        geom: GeometryModel3D,
        ds3d_meta,
        cube_volume: np.ndarray,
        inlines_3d: np.ndarray,
        xlines_3d: np.ndarray,
    ) -> tuple[CalibrationPair, CalibrationPair]:
        """Build train/test calibration pairs from overlap zones."""
        cfg = self._cfg
        all_coords, all_2d, all_3d = [], [], []

        n_il = len(inlines_3d)
        n_xl = len(xlines_3d)

        for line in lines:
            inside_idx = overlap.overlap_indices(line.coords)
            if len(inside_idx) == 0:
                logger.warning("Line '{}' has no overlap with 3D", line.name)
                continue

            for idx in inside_idx:
                x, y = line.coords[idx]
                # Find nearest 3D trace via geometry
                il_frac, xl_frac = geom.xy_to_ilxl(
                    np.array([x]), np.array([y])
                )
                il_near = int(round(il_frac[0]))
                xl_near = int(round(xl_frac[0]))

                # Map to volume indices
                il_idx_arr = np.searchsorted(inlines_3d, il_near)
                xl_idx_arr = np.searchsorted(xlines_3d, xl_near)
                if il_idx_arr >= n_il or xl_idx_arr >= n_xl:
                    continue
                if inlines_3d[il_idx_arr] != il_near or xlines_3d[xl_idx_arr] != xl_near:
                    # Use bilinear interpolation between neighbouring traces
                    trace_3d = self._bilinear_3d(
                        cube_volume, inlines_3d, xlines_3d, il_frac[0], xl_frac[0]
                    )
                else:
                    trace_3d = cube_volume[il_idx_arr, xl_idx_arr]

                all_coords.append([x, y])
                all_2d.append(line.data[idx])
                all_3d.append(trace_3d)

        if not all_2d:
            logger.error("No calibration pairs found!")
            empty = CalibrationPair(
                coords=np.empty((0, 2)),
                amp_2d=np.empty((0, ds3d_meta.n_samples)),
                amp_3d=np.empty((0, ds3d_meta.n_samples)),
                dt_ms=ds3d_meta.dt_ms,
            )
            return empty, empty

        coords_arr = np.array(all_coords)
        amp_2d_arr = np.array(all_2d, dtype=np.float32)
        amp_3d_arr = np.array(all_3d, dtype=np.float32)

        n = len(amp_2d_arr)
        logger.info("Total calibration pairs: {}", n)

        # Spatial holdout split
        holdout = cfg.calibration.holdout_fraction
        if cfg.calibration.holdout_mode == "segment":
            # Split by contiguous segments (by index)
            n_test = max(1, int(n * holdout))
            # Take the last segment as test
            train_idx = np.arange(0, n - n_test)
            test_idx = np.arange(n - n_test, n)
        else:
            # Holdout entire lines: simplified — last fraction of pairs
            n_test = max(1, int(n * holdout))
            train_idx = np.arange(0, n - n_test)
            test_idx = np.arange(n - n_test, n)

        train = CalibrationPair(
            coords=coords_arr[train_idx],
            amp_2d=amp_2d_arr[train_idx],
            amp_3d=amp_3d_arr[train_idx],
            dt_ms=ds3d_meta.dt_ms,
        )
        test = CalibrationPair(
            coords=coords_arr[test_idx],
            amp_2d=amp_2d_arr[test_idx],
            amp_3d=amp_3d_arr[test_idx],
            dt_ms=ds3d_meta.dt_ms,
        )
        logger.info("Train: {} pairs, Test: {} pairs", len(train_idx), len(test_idx))
        return train, test

    @staticmethod
    def _bilinear_3d(
        volume: np.ndarray,
        inlines: np.ndarray,
        xlines: np.ndarray,
        il_frac: float,
        xl_frac: float,
    ) -> np.ndarray:
        """Bilinear interpolation of a 3D trace at fractional (il, xl)."""
        il_idx = np.interp(il_frac, inlines, np.arange(len(inlines)))
        xl_idx = np.interp(xl_frac, xlines, np.arange(len(xlines)))

        i0 = int(np.floor(il_idx))
        i1 = min(i0 + 1, len(inlines) - 1)
        j0 = int(np.floor(xl_idx))
        j1 = min(j0 + 1, len(xlines) - 1)

        di = il_idx - i0
        dj = xl_idx - j0

        trace = (
            volume[i0, j0] * (1 - di) * (1 - dj)
            + volume[i1, j0] * di * (1 - dj)
            + volume[i0, j1] * (1 - di) * dj
            + volume[i1, j1] * di * dj
        )
        return trace.astype(np.float32)

    @staticmethod
    def _create_simulation_mask(
        orig_mask: np.ndarray,
        sparse_mask: np.ndarray,
        inlines_3d: np.ndarray,
        xlines_3d: np.ndarray,
        grid,
    ) -> np.ndarray:
        """Create a mask inside the original 3D that simulates 2D line geometry.

        Takes the pattern of sparse observations from 2D lines and projects it
        onto the 3D area.  Where 2D lines cross the 3D, those positions are
        "observed"; everything else in the 3D area is "missing" (to be
        reconstructed).
        """
        sim_mask = np.zeros_like(orig_mask)
        # Where both original and sparse data exist → simulated observation
        sim_mask = orig_mask & sparse_mask
        # If too few observations, fall back to random sub-sampling
        if sim_mask.sum() < 10:
            logger.warning("Few simulation observations; using random 10% of 3D traces")
            rng = np.random.default_rng(42)
            sim_mask = orig_mask.copy()
            orig_positions = np.argwhere(orig_mask)
            n_keep = max(1, int(0.1 * len(orig_positions)))
            keep = rng.choice(len(orig_positions), n_keep, replace=False)
            sim_mask[:] = False
            for k in keep:
                sim_mask[orig_positions[k, 0], orig_positions[k, 1]] = True
        return sim_mask
