"""
seis2cube — Streamlit Dashboard

Launch with:
    streamlit run seis2cube/ui/app.py
    # or
    seis2cube ui --config config.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="seis2cube",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from seis2cube.ui.state import (
    PipelineStage,
    STAGE_LABELS,
    STAGE_ORDER,
    WorkStep,
    STEP_LABELS,
    add_log,
    init_state,
    set_stage,
)
from seis2cube.ui.components import (
    PALETTE,
    SEISMIC_COLORSCALE,
    plot_convergence,
    plot_crossline_section,
    plot_inline_section,
    plot_map_with_lines,
    plot_metrics_radar,
    plot_spectrum_comparison,
    plot_time_slice,
    plot_trace_comparison,
    render_metric_card,
    render_pipeline_progress,
    render_status_badge,
    section_header,
)

init_state()

# ── Native Streamlit Interface ─────────────────────────────────────────

# ── Sidebar ─────────────────────────────────────────────────────────────


def _sidebar() -> None:
    with st.sidebar:
        st.title("🌊 seis2cube")
        st.caption("3D SEG-Y Extension from 2D Profiles")
        st.divider()

        # ── Data paths ─────────────────────────────────────────────────
        st.header("📂 Input Data")

        # Auto-load from query param ?config=path/to/config.yaml
        _qp = st.query_params
        if "config" in _qp and st.session_state.config_obj is None:
            _cfg_path = Path(_qp["config"])
            if _cfg_path.exists():
                import yaml
                try:
                    raw = yaml.safe_load(_cfg_path.read_text())
                    from seis2cube.config import PipelineConfig
                    st.session_state.config_dict = raw
                    st.session_state.config_obj = PipelineConfig(**raw)
                except Exception:
                    pass

        # Determine defaults from existing config, session, or hardcoded test paths
        _DEFAULT_3D_DIR = "test_data2"
        _DEFAULT_2D_DIR = "test_data2"

        _default_3d = st.session_state.get("_ui_3d_dir", "")
        _default_2d = st.session_state.get("_ui_2d_dir", "")
        if not _default_3d and st.session_state.config_obj is not None:
            _default_3d = str(st.session_state.config_obj.cube3d_path.parent)
        if not _default_2d and st.session_state.config_obj is not None and st.session_state.config_obj.lines2d_paths:
            _default_2d = str(st.session_state.config_obj.lines2d_paths[0].parent)
        if not _default_3d:
            _default_3d = _DEFAULT_3D_DIR
        if not _default_2d:
            _default_2d = _DEFAULT_2D_DIR

        cube3d_dir = st.text_input(
            "📦 3D cube folder",
            value=_default_3d,
            placeholder="/path/to/3d_folder",
            help="Folder with one 3D SEG-Y file",
        )
        lines2d_dir = st.text_input(
            "📏 2D profiles folder",
            value=_default_2d,
            placeholder="/path/to/2d_folder",
            help="Folder with 2D SEG-Y profiles",
        )
        st.session_state["_ui_3d_dir"] = cube3d_dir
        st.session_state["_ui_2d_dir"] = lines2d_dir

        # ── Auto-detect SEG-Y files ───────────────────────────────────
        _SEGY_EXT = {".segy", ".sgy", ".seg"}
        _SKIP_EXT = {".json", ".yaml", ".yml", ".txt", ".csv", ".py",
                     ".md", ".geojson", ".shp", ".xml", ".html", ".png",
                     ".jpg", ".pdf", ".log"}

        def _find_cube_file(folder: str) -> list[Path]:
            """Find 3D cube: files with .segy/.sgy extension."""
            p = Path(folder)
            if not p.is_dir():
                return []
            return sorted(f for f in p.iterdir()
                          if f.is_file() and f.suffix.lower() in _SEGY_EXT)

        def _find_2d_files(folder: str, exclude: list[Path] | None = None) -> list[Path]:
            """Find 2D profiles: all seismic files EXCEPT known 3D cubes."""
            p = Path(folder)
            if not p.is_dir():
                return []
            excl = {f.resolve() for f in (exclude or [])}
            result = []
            for f in sorted(p.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix.lower() in _SKIP_EXT:
                    continue
                if f.resolve() in excl:
                    continue
                result.append(f)
            return result

        _cube_files: list[Path] = []
        _line_files: list[Path] = []
        if cube3d_dir:
            _cube_files = _find_cube_file(cube3d_dir)
        if lines2d_dir:
            _line_files = _find_2d_files(lines2d_dir, exclude=_cube_files)

        # Show what was found
        if cube3d_dir and not _cube_files:
            st.warning("No SEG-Y in 3D folder")
        if lines2d_dir and not _line_files:
            st.warning("No SEG-Y in 2D folder")
        if _cube_files and _line_files:
            st.caption(f"✓ 3D: **{_cube_files[0].name}** · 2D: **{len(_line_files)}** profiles")

        # ── Step Navigation ───────────────────────────────────────────────
        st.divider()
        st.markdown("**📋 Workflow Steps**")

        current_step = st.session_state.work_step

        # Show step status indicators
        for step in [WorkStep.DATA_LOADING, WorkStep.CALIBRATION, WorkStep.INTERPOLATION, WorkStep.DONE]:
            label = STEP_LABELS[step]
            if step == current_step:
                st.markdown(f"▶ **{label}** ← current")
            elif step < current_step:
                st.markdown(f"✅ ~~{label}~~")
            else:
                st.markdown(f"○ {label}")

        # Step jump buttons
        st.caption("Navigate to step:")
        nav_cols = st.columns(3)
        with nav_cols[0]:
            if st.button("1️⃣ Data", disabled=current_step == WorkStep.DATA_LOADING, use_container_width=True):
                st.session_state.work_step = WorkStep.DATA_LOADING
                st.rerun()
        with nav_cols[1]:
            can_cal = st.session_state.data_loaded
            if st.button("2️⃣ Cal", disabled=not can_cal or current_step == WorkStep.CALIBRATION, use_container_width=True):
                st.session_state.work_step = WorkStep.CALIBRATION
                st.rerun()
        with nav_cols[2]:
            can_interp = st.session_state.calibration_done
            if st.button("3️⃣ Interp", disabled=not can_interp or current_step == WorkStep.INTERPOLATION, use_container_width=True):
                st.session_state.work_step = WorkStep.INTERPOLATION
                st.rerun()

        st.divider()

        # ── Step-Specific Settings ─────────────────────────────────────
        if current_step == WorkStep.DATA_LOADING:
            st.markdown("**📁 Step 1: Data Loading Settings**")

            expand_pct = st.slider(
                "Expansion buffer, % of 3D size",
                min_value=0, max_value=200, value=50, step=5,
                help="Buffer around 3D hull as % of mean side.",
            )

            test_split_pct = st.slider(
                "Test split, % of overlap data",
                min_value=5, max_value=50, value=20, step=5,
                help="Percentage of overlap zone held out for testing calibration quality.",
            )
            st.session_state.test_split_pct = test_split_pct

            # Build base config
            if _cube_files and _line_files:
                try:
                    from seis2cube.config import PipelineConfig
                    cfg_obj = PipelineConfig(
                        cube3d_path=_cube_files[0],
                        lines2d_paths=_line_files,
                        expand_polygon_path=None,
                        expand_buffer_pct=float(expand_pct),
                    )
                    st.session_state.config_obj = cfg_obj
                    st.success("✅ Data configured")
                except Exception as e:
                    st.error(f"Config error: {e}")

            # Load button
            if st.button("📥 Load Data", type="primary", use_container_width=True,
                        disabled=not (_cube_files and _line_files)):
                _run_step_data_loading()

        elif current_step == WorkStep.CALIBRATION:
            st.markdown("**🔧 Step 2: Calibration Settings**")

            cal_opts = ["global_shift", "windowed", "linear_regression", "gbdt"]
            cal_method = st.selectbox("Method", cal_opts,
                                      index=cal_opts.index(st.session_state.get("cal_method", "global_shift")),
                                      key="cal_method_select")
            st.session_state.cal_method = cal_method

            # Method-specific parameters
            with st.expander("⚙️ Method Parameters", expanded=True):
                cal_params = {}

                if cal_method == "global_shift":
                    cal_params["max_shift_ms"] = st.slider("Max time shift (ms)", 10.0, 200.0, 50.0, 5.0)
                    cal_params["estimate_phase"] = st.checkbox("Estimate phase rotation", value=True)
                    cal_params["estimate_matching_filter"] = st.checkbox("Use spectral matching", value=False)
                    if cal_params["estimate_matching_filter"]:
                        cal_params["matching_filter_taps"] = st.slider("Filter taps", 11, 101, 51, 2)

                elif cal_method == "windowed":
                    cal_params["window_ms"] = st.slider("Window length (ms)", 100.0, 1000.0, 400.0, 50.0)
                    cal_params["overlap_ms"] = st.slider("Window overlap (ms)", 0.0, 200.0, 100.0, 10.0)
                    cal_params["max_shift_ms"] = st.slider("Max shift per window (ms)", 10.0, 100.0, 30.0, 5.0)
                    cal_params["cc_threshold"] = st.slider("Min correlation threshold", 0.1, 0.9, 0.3, 0.05)

                elif cal_method == "linear_regression":
                    reg_opts = ["ridge", "elasticnet", "huber"]
                    cal_params["regressor_name"] = st.selectbox("Regressor type", reg_opts, index=0)
                    cal_params["window_samples"] = st.slider("Feature window (samples)", 16, 256, 64, 8)

                elif cal_method == "gbdt":
                    backend_opts = ["sklearn", "xgboost", "lightgbm"]
                    cal_params["backend"] = st.selectbox("GBDT backend", backend_opts, index=0)
                    cal_params["n_estimators"] = st.slider("Number of trees", 50, 500, 200, 25)
                    cal_params["max_depth"] = st.slider("Max tree depth", 3, 12, 6, 1)
                    cal_params["learning_rate"] = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
                    cal_params["window_samples"] = st.slider("Feature window (samples)", 16, 256, 64, 8)

                st.session_state.cal_params = cal_params

            if st.button("🎯 Run Calibration", type="primary", use_container_width=True,
                        disabled=not st.session_state.data_loaded):
                _run_step_calibration()

        elif current_step == WorkStep.INTERPOLATION:
            st.markdown("**🧩 Step 3: Interpolation Settings**")

            interp_opts = ["idw", "pocs", "mssa"]
            interp_method = st.selectbox("Method", interp_opts,
                                         index=interp_opts.index(st.session_state.get("interp_method", "idw")),
                                         key="interp_method_select")
            st.session_state.interp_method = interp_method

            # Method-specific parameters
            with st.expander("⚙️ Method Parameters", expanded=True):
                interp_params = {}

                if interp_method == "idw":
                    interp_params["power"] = st.slider("IDW power", 0.5, 4.0, 2.0, 0.5)
                    interp_params["max_neighbours"] = st.slider("Max neighbours", 4, 32, 12, 2)

                elif interp_method == "pocs":
                    interp_params["n_iter"] = st.slider("Iterations", 20, 300, 100, 10)
                    interp_params["fast"] = st.checkbox("Use FPOCS (fast)", value=True)
                    interp_params["threshold_start_pct"] = st.slider("Start threshold (%)", 90.0, 99.9, 99.0, 0.1)
                    interp_params["threshold_end_pct"] = st.slider("End threshold (%)", 0.1, 10.0, 1.0, 0.1)

                elif interp_method == "mssa":
                    interp_params["rank"] = st.slider("Target rank", 5, 100, 20, 5)
                    interp_params["window"] = st.slider("Window length (traces)", 10, 200, 50, 10)
                    interp_params["n_iter"] = st.slider("Iterations", 5, 100, 30, 5)

                st.session_state.interp_params = interp_params

            blend_traces = st.slider("Blend taper (traces)", 0, 50, 10, 1,
                                     help="Cosine taper at seam between original and extended zones.")
            st.session_state.blend_taper = blend_traces

            if st.button("🚀 Run Interpolation", type="primary", use_container_width=True,
                        disabled=not st.session_state.calibration_done):
                _run_step_interpolation()

        # Global reset
        st.divider()
        if st.button("↺ Reset All", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        # ── Log ─────────────────────────────────────────────────────────
        if st.session_state.pipeline_log:
            st.divider()
            with st.expander("📋 Pipeline Log", expanded=False):
                for msg in st.session_state.pipeline_log[-30:]:
                    st.text(msg)


# ── Step-by-step pipeline runners ───────────────────────────────────────


def _create_calibrator_with_params(cfg):
    """Create calibrator with UI-selected parameters."""
    from seis2cube.calibration.global_shift import GlobalShiftGainPhase
    from seis2cube.calibration.windowed import WindowedShiftGain
    from seis2cube.calibration.regression import LinearRegressionCalibrator
    from seis2cube.calibration.gbdt import GBDTCalibrator

    cal_method = st.session_state.cal_method
    cal_params = st.session_state.cal_params

    if cal_method == "global_shift":
        return GlobalShiftGainPhase(
            max_shift_ms=cal_params.get("max_shift_ms", 50.0),
            estimate_phase=cal_params.get("estimate_phase", True),
            estimate_matching_filter=cal_params.get("estimate_matching_filter", False),
            matching_filter_taps=cal_params.get("matching_filter_taps", 51),
        )
    elif cal_method == "windowed":
        return WindowedShiftGain(
            window_ms=cal_params.get("window_ms", 400.0),
            overlap_ms=cal_params.get("overlap_ms", 100.0),
            max_shift_ms=cal_params.get("max_shift_ms", 30.0),
            cc_threshold=cal_params.get("cc_threshold", 0.3),
        )
    elif cal_method == "linear_regression":
        return LinearRegressionCalibrator(
            regressor_name=cal_params.get("regressor_name", "ridge"),
            window_samples=cal_params.get("window_samples", 64),
        )
    elif cal_method == "gbdt":
        return GBDTCalibrator(
            n_estimators=cal_params.get("n_estimators", 200),
            max_depth=cal_params.get("max_depth", 6),
            learning_rate=cal_params.get("learning_rate", 0.1),
            backend=cal_params.get("backend", "sklearn"),
            window_samples=cal_params.get("window_samples", 64),
        )
    else:
        raise ValueError(f"Unknown calibration method: {cal_method}")


def _create_interpolator_with_params():
    """Create interpolator with UI-selected parameters."""
    from seis2cube.interpolation.idw import IDWTimeSliceInterpolator
    from seis2cube.interpolation.pocs import POCSInterpolator
    from seis2cube.interpolation.mssa import MSSAInterpolator

    interp_method = st.session_state.interp_method
    interp_params = st.session_state.interp_params

    if interp_method == "idw":
        return IDWTimeSliceInterpolator(
            power=interp_params.get("power", 2.0),
            max_neighbours=interp_params.get("max_neighbours", 12),
        )
    elif interp_method == "pocs":
        return POCSInterpolator(
            n_iter=interp_params.get("n_iter", 100),
            fast=interp_params.get("fast", True),
            threshold_start_pct=interp_params.get("threshold_start_pct", 99.0),
            threshold_end_pct=interp_params.get("threshold_end_pct", 1.0),
        )
    elif interp_method == "mssa":
        return MSSAInterpolator(
            rank=interp_params.get("rank", 20),
            window=interp_params.get("window", 50),
            n_iter=interp_params.get("n_iter", 30),
        )
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")


def _run_step_data_loading() -> None:
    """Step 1: Load 3D cube, 2D lines, build geometry and overlap detection."""
    cfg = st.session_state.config_obj
    if cfg is None:
        st.error("No config loaded")
        return

    add_log("Starting Step 1: Data Loading...")

    try:
        from seis2cube.io.segy_dataset import SegyDataset
        from seis2cube.geometry.crs_converter import CRSConverter
        from seis2cube.geometry.geometry_model import AffineGridMapper, KDTreeMapper
        from seis2cube.geometry.overlap_detector import OverlapDetector
        from seis2cube.pipeline.runner import PipelineRunner

        set_stage(PipelineStage.INGESTING, 0.1)
        runner = PipelineRunner(cfg)

        # 1. Ingest 3D
        with SegyDataset(cfg.cube3d_path, cfg.header_bytes, cfg.io) as ds3d:
            meta3d = ds3d.meta
            st.session_state.meta_3d = {
                "n_traces": meta3d.n_traces,
                "n_samples": meta3d.n_samples,
                "dt_ms": meta3d.dt_ms,
                "delrt_ms": meta3d.delrt_ms,
                "format": meta3d.sample_format,
                "n_inlines": len(meta3d.inlines) if meta3d.inlines is not None else 0,
                "n_xlines": len(meta3d.xlines) if meta3d.xlines is not None else 0,
            }
            coords_3d = ds3d.all_coordinates()
            cube_volume = runner._load_3d_volume(ds3d)
            inlines_3d = meta3d.inlines
            xlines_3d = meta3d.xlines

        st.session_state.cube_volume = cube_volume
        st.session_state.coords_3d = coords_3d
        st.session_state.inlines_3d = inlines_3d
        st.session_state.xlines_3d = xlines_3d
        add_log(f"Loaded 3D: {meta3d.n_traces} traces, {len(inlines_3d)}×{len(xlines_3d)} IL×XL")

        # 2. CRS
        set_stage(PipelineStage.CRS_CONVERSION, 0.25)
        crs_conv = CRSConverter(cfg.crs)
        st.session_state.crs_conv = crs_conv
        if not crs_conv.is_identity:
            cx, cy = crs_conv.forward(coords_3d[:, 0], coords_3d[:, 1])
            coords_3d = np.column_stack([cx, cy])
            st.session_state.coords_3d = coords_3d
            add_log("CRS conversion applied")

        # 3. Geometry
        set_stage(PipelineStage.GEOMETRY, 0.4)
        if inlines_3d is not None and xlines_3d is not None:
            geom = AffineGridMapper(coords_3d, inlines_3d, xlines_3d)
        else:
            ilxl = np.zeros((len(coords_3d), 2))
            geom = KDTreeMapper(coords_3d, ilxl[:, 0], ilxl[:, 1])
        st.session_state.geom = geom
        add_log("Geometry model built")

        # 4. Overlap
        set_stage(PipelineStage.OVERLAP, 0.55)
        if cfg.expand_polygon_path is not None:
            expand_poly = OverlapDetector.load_polygon(cfg.expand_polygon_path)
            if not crs_conv.is_identity:
                from shapely.ops import transform as shp_transform
                expand_poly = shp_transform(
                    lambda x, y: crs_conv.forward(np.array(x), np.array(y)), expand_poly
                )
        else:
            expand_poly = OverlapDetector.auto_expand_polygon(
                coords_3d, buffer_pct=cfg.expand_buffer_pct,
            )
        overlap = OverlapDetector.from_3d_coords(coords_3d, expand_polygon=expand_poly)
        st.session_state.expand_poly = expand_poly
        st.session_state.overlap = overlap
        add_log(f"Expansion polygon created, overlap zones detected")

        # 5. Load 2D
        set_stage(PipelineStage.LOADING_2D, 0.8)
        lines_2d = []
        lines_info = []
        for lpath in cfg.lines2d_paths:
            line = runner._load_2d_line(lpath, crs_conv, meta3d.dt_ms, meta3d.n_samples,
                                             target_delrt_ms=meta3d.delrt_ms)
            lines_2d.append(line)
            lines_info.append({"name": line.name, "n_traces": line.n_traces, "dt_ms": line.dt_ms})
        st.session_state.lines_2d = lines_2d
        st.session_state.lines_info = lines_info
        st.session_state.lines_coords = [l.coords for l in lines_2d]
        st.session_state.line_names = [l.name for l in lines_2d]
        st.session_state.runner = runner  # Store for later steps
        st.session_state.meta3d = meta3d  # Store for later steps

        add_log(f"Loaded {len(lines_2d)} 2D profiles")

        # Complete
        st.session_state.data_loaded = True
        st.session_state.work_step = WorkStep.CALIBRATION
        set_stage(PipelineStage.IDLE, 0.0)
        add_log("Step 1 complete! Proceeding to Calibration.")
        st.rerun()

    except Exception as e:
        set_stage(PipelineStage.ERROR)
        st.session_state.error_msg = str(e)
        add_log(f"ERROR in Data Loading: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Data loading failed: {e}")


def _run_step_calibration() -> None:
    """Step 2: Run calibration with user-selected method and parameters."""
    if not st.session_state.data_loaded:
        st.error("Please load data first (Step 1)")
        return

    add_log("Starting Step 2: Calibration...")

    try:
        set_stage(PipelineStage.CALIBRATION, 0.3)

        # Get stored data
        lines_2d = st.session_state.lines_2d
        overlap = st.session_state.overlap
        geom = st.session_state.geom
        meta3d = st.session_state.meta3d
        cube_volume = st.session_state.cube_volume
        inlines_3d = st.session_state.inlines_3d
        xlines_3d = st.session_state.xlines_3d
        runner = st.session_state.runner
        test_split_pct = st.session_state.test_split_pct

        # Build calibration pairs with configurable test split
        train_pairs, test_pairs = runner._build_calibration_pairs(
            lines_2d, overlap, geom, meta3d, cube_volume, inlines_3d, xlines_3d,
            test_ratio=test_split_pct / 100.0,
        )

        add_log(f"Calibration pairs: {len(train_pairs.amp_2d)} train, {len(test_pairs.amp_2d)} test")

        # Create calibrator with UI parameters
        calibrator = _create_calibrator_with_params(st.session_state.config_obj)
        cal_model = calibrator.fit(train_pairs)
        st.session_state.cal_model = cal_model
        st.session_state.calibrator = calibrator

        add_log(f"Calibration model fitted: {st.session_state.cal_method}")

        # Evaluate on test set
        if test_pairs.amp_2d.shape[0] > 0:
            metrics_after = calibrator.evaluate(test_pairs, cal_model)
            raw = test_pairs.amp_2d
            ref = test_pairs.amp_3d
            corrs_raw = [float(np.corrcoef(raw[i], ref[i])[0, 1])
                         for i in range(len(raw)) if np.isfinite(np.corrcoef(raw[i], ref[i])[0, 1])]
            baseline_corr = np.mean(corrs_raw) if corrs_raw else 0.0
            baseline_rmse = float(np.sqrt(np.mean((raw - ref) ** 2)))

            st.session_state.cal_metrics_before = {"corr": float(baseline_corr), "rmse": baseline_rmse}
            st.session_state.cal_metrics_after = metrics_after

            # Store test traces
            n_test = len(test_pairs.amp_2d)
            st.session_state.sample_traces_2d = test_pairs.amp_2d
            st.session_state.sample_traces_3d = test_pairs.amp_3d
            corrected_test = calibrator._apply_array(test_pairs.amp_2d, cal_model)
            st.session_state.sample_traces_corr = corrected_test
            st.session_state.dt_ms = meta3d.dt_ms
            st.session_state.n_train = len(train_pairs.amp_2d)
            st.session_state.n_test = n_test

            # Per-pair stats
            per_corr_before = []
            per_corr_after = []
            for i in range(n_test):
                cc_b = float(np.corrcoef(test_pairs.amp_2d[i], test_pairs.amp_3d[i])[0, 1])
                cc_a = float(np.corrcoef(corrected_test[i], test_pairs.amp_3d[i])[0, 1])
                per_corr_before.append(cc_b if np.isfinite(cc_b) else 0.0)
                per_corr_after.append(cc_a if np.isfinite(cc_a) else 0.0)
            st.session_state.per_corr_before = np.array(per_corr_before)
            st.session_state.per_corr_after = np.array(per_corr_after)

            add_log(f"Test metrics - Before: corr={baseline_corr:.3f}, rmse={baseline_rmse:.1f}")
            add_log(f"Test metrics - After: corr={metrics_after.get('corr', 0):.3f}, rmse={metrics_after.get('rmse', 0):.1f}")

        # Apply calibration to all lines
        set_stage(PipelineStage.APPLYING_CAL, 0.7)
        calibrated_lines = [calibrator.apply(line, cal_model) for line in lines_2d]
        st.session_state.calibrated_lines = calibrated_lines

        # Build target grid
        set_stage(PipelineStage.BUILDING_GRID, 0.9)
        from seis2cube.pipeline.volume_builder import VolumeBuilder

        n_il_orig = len(inlines_3d)
        n_xl_orig = len(xlines_3d)
        coords_3d = st.session_state.coords_3d
        p00 = coords_3d[0]
        p0n = coords_3d[n_xl_orig - 1]
        pn0 = coords_3d[(n_il_orig - 1) * n_xl_orig]
        il_step_xy = (pn0 - p00) / max(n_il_orig - 1, 1)
        xl_step_xy = (p0n - p00) / max(n_xl_orig - 1, 1)
        expand_poly = st.session_state.expand_poly

        vb = VolumeBuilder(
            geometry=geom, orig_inlines=inlines_3d, orig_xlines=xlines_3d,
            n_samples=meta3d.n_samples, dt_ms=meta3d.dt_ms, expand_polygon=expand_poly,
            origin_x=float(p00[0]), origin_y=float(p00[1]),
            il_step_x=float(il_step_xy[0]), il_step_y=float(il_step_xy[1]),
            xl_step_x=float(xl_step_xy[0]), xl_step_y=float(xl_step_xy[1]),
        )
        target_grid = vb.build_target_grid(max_volume_gb=st.session_state.config_obj.max_grid_memory_gb)
        st.session_state.target_grid = target_grid

        # Inject data
        sparse = vb.inject_lines(target_grid, calibrated_lines)
        orig_in_grid, orig_mask = vb.inject_original_3d(target_grid, cube_volume, inlines_3d, xlines_3d)

        st.session_state.orig_mask = orig_mask
        st.session_state.sparse_mask = sparse.mask.copy()
        st.session_state.sparse_fill = float(sparse.fill_ratio)
        st.session_state.orig_in_grid = orig_in_grid
        st.session_state.vb = vb

        add_log(f"Target grid: {target_grid.n_il}×{target_grid.n_xl} IL×XL")

        # Complete
        st.session_state.calibration_done = True
        st.session_state.work_step = WorkStep.INTERPOLATION
        set_stage(PipelineStage.IDLE, 0.0)
        add_log("Step 2 complete! Proceeding to Interpolation.")
        st.rerun()

    except Exception as e:
        set_stage(PipelineStage.ERROR)
        st.session_state.error_msg = str(e)
        add_log(f"ERROR in Calibration: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Calibration failed: {e}")


def _run_step_interpolation() -> None:
    """Step 3: Run interpolation with user-selected method and parameters."""
    if not st.session_state.calibration_done:
        st.error("Please run calibration first (Step 2)")
        return

    add_log("Starting Step 3: Interpolation...")

    try:
        # Get stored data
        orig_in_grid = st.session_state.orig_in_grid
        orig_mask = st.session_state.orig_mask
        sparse_mask = st.session_state.sparse_mask
        target_grid = st.session_state.target_grid
        vb = st.session_state.vb
        inlines_3d = st.session_state.inlines_3d
        xlines_3d = st.session_state.xlines_3d
        runner = st.session_state.runner
        cfg = st.session_state.config_obj

        # Create interpolator with UI parameters
        set_stage(PipelineStage.TUNING_INTERP, 0.3)
        interpolator = _create_interpolator_with_params()

        # Simulation mask for quality check
        sim_mask = runner._create_simulation_mask(orig_mask, sparse_mask, inlines_3d, xlines_3d, target_grid)
        sim_metrics = interpolator.fit(orig_in_grid, sim_mask)
        st.session_state.interp_sim_metrics = sim_metrics
        st.session_state.interp_sim_mask = sim_mask

        add_log(f"Simulation metrics: RMSE={sim_metrics.get('rmse', 0):.1f}, MAE={sim_metrics.get('mae', 0):.1f}")

        # Reconstruct
        set_stage(PipelineStage.RECONSTRUCTING, 0.6)

        # Get sparse volume with calibrated lines
        calibrated_lines = st.session_state.calibrated_lines
        sparse = vb.inject_lines(target_grid, calibrated_lines)

        combined_data = orig_in_grid.copy()
        combined_mask = orig_mask.copy()
        ext_only = sparse.mask & ~orig_mask
        combined_data[ext_only] = np.nan_to_num(sparse.data[ext_only], nan=0.0)
        combined_mask[ext_only] = True

        from seis2cube.models.volume import SparseVolume as SV
        combined_sparse = SV(grid=target_grid, data=combined_data, mask=combined_mask)
        result = interpolator.reconstruct(combined_sparse)

        st.session_state.recon_volume = result.volume
        add_log("Interpolation reconstruction complete")

        # Scatter plot data
        if sim_mask is not None and orig_in_grid is not None:
            sim_positions = np.argwhere(sim_mask & ~orig_mask)
            if len(sim_positions) > 0:
                n_samples = min(len(sim_positions), 3000)
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(sim_positions), n_samples, replace=False)
                sampled_pos = sim_positions[sample_idx]

                true_vals, pred_vals = [], []
                for il_idx, xl_idx in sampled_pos:
                    true_trace = orig_in_grid[il_idx, xl_idx, :]
                    pred_trace = result.volume[il_idx, xl_idx, :]
                    true_vals.append(np.sqrt(np.mean(true_trace**2)))
                    pred_vals.append(np.sqrt(np.mean(pred_trace**2)))

                st.session_state.interp_scatter_true = np.array(true_vals)
                st.session_state.interp_scatter_pred = np.array(pred_vals)

        # Assemble with blending
        set_stage(PipelineStage.ASSEMBLING, 0.8)
        blend_taper = st.session_state.get("blend_taper", 10)
        final = VolumeBuilder.assemble(
            orig_vol=orig_in_grid, orig_mask=orig_mask, recon_vol=result.volume,
            taper_width=blend_taper,
            blend=True,
        )
        st.session_state.final_volume = final
        add_log(f"Volume assembled with {blend_taper} trace taper")

        # Write output
        set_stage(PipelineStage.WRITING, 0.9)
        from seis2cube.io.segy_writer import SegyWriter3D

        meta3d = st.session_state.meta3d
        writer = SegyWriter3D(
            path=cfg.out_cube_path, inlines=target_grid.inlines, xlines=target_grid.xlines,
            dt_us=int(meta3d.dt_ms * 1000), header_bytes=cfg.header_bytes,
            origin_x=target_grid.origin_x, origin_y=target_grid.origin_y,
            il_step_x=target_grid.il_step_x, il_step_y=target_grid.il_step_y,
            xl_step_x=target_grid.xl_step_x, xl_step_y=target_grid.xl_step_y,
            delrt_ms=meta3d.delrt_ms,
        )
        out_path = writer.write(final)

        # QC Report
        set_stage(PipelineStage.QC, 0.95)
        from seis2cube.qc.reporter import QCReporter

        qc = QCReporter(cfg.qc_report_dir)
        if st.session_state.cal_metrics_before:
            qc.log_calibration(
                st.session_state.cal_metrics_before["corr"],
                st.session_state.cal_metrics_before["rmse"],
                st.session_state.cal_metrics_after or {},
            )
        if sim_metrics:
            qc.log_interpolation_sim(sim_metrics)
        qc.save(cfg)

        # Complete
        st.session_state.interpolation_done = True
        st.session_state.work_step = WorkStep.DONE
        set_stage(PipelineStage.DONE, 1.0)
        add_log(f"Step 3 complete! Output: {out_path}")
        st.success(f"✅ Pipeline complete! Output saved to: {out_path}")

    except Exception as e:
        set_stage(PipelineStage.ERROR)
        st.session_state.error_msg = str(e)
        add_log(f"ERROR in Interpolation: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Interpolation failed: {e}")


# ── Main content area ──────────────────────────────────────────────────

tab_dash, tab_data, tab_cal, tab_interp, tab_viewer, tab_qc = st.tabs([
    "📊 Dashboard",
    "📁 Data Explorer",
    "🔧 Calibration",
    "🧩 Interpolation",
    "🗺️ Volume Viewer",
    "📈 QC Report",
])

_sidebar()


# ════════════════════════════════════════════════════════════════════════
# TAB: Dashboard
# ════════════════════════════════════════════════════════════════════════

with tab_dash:
    section_header("Pipeline Overview", "📊")

    stage = st.session_state.pipeline_stage
    if stage not in (PipelineStage.IDLE,):
        idx = STAGE_ORDER.index(stage) + 1 if stage in STAGE_ORDER else (
            len(STAGE_ORDER) if stage == PipelineStage.DONE else 0
        )
        render_pipeline_progress(
            STAGE_LABELS[stage],
            st.session_state.pipeline_progress,
            idx,
            len(STAGE_ORDER),
        )

    if stage == PipelineStage.ERROR:
        st.error(f"Pipeline error: {st.session_state.error_msg}")

    # Metric cards
    meta = st.session_state.meta_3d
    if meta is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("3D Traces", f"{meta['n_traces']:,}", icon="📦")
        with c2:
            render_metric_card("Samples", str(meta["n_samples"]), icon="📏")
        with c3:
            render_metric_card("Δt", f"{meta['dt_ms']:.1f} ms", icon="⏱️")
        with c4:
            n_lines = len(st.session_state.lines_info)
            render_metric_card("2D Lines", str(n_lines), icon="📐")

        st.markdown("<br>", unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            render_metric_card("Inlines", str(meta["n_inlines"]), icon="↔️")
        with c6:
            render_metric_card("Crosslines", str(meta["n_xlines"]), icon="↕️")
        with c7:
            cal_after = st.session_state.cal_metrics_after
            if cal_after:
                render_metric_card(
                    "Cal. Correlation",
                    f"{cal_after['pearson_corr']:.3f}",
                    icon="🎯",
                )
            else:
                render_metric_card("Cal. Correlation", "—", icon="🎯")
        with c8:
            sim = st.session_state.interp_sim_metrics
            if sim:
                render_metric_card(
                    "Interp. RMSE",
                    f"{sim['rmse']:.4f}",
                    icon="📐",
                )
            else:
                render_metric_card("Interp. RMSE", "—", icon="📐")
    else:
        st.info("Set data paths in the sidebar and run the pipeline to see results.")

    # Config summary
    if st.session_state.config_obj is not None:
        cfg = st.session_state.config_obj
        with st.expander("Active Configuration", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**3D Cube:** `{cfg.cube3d_path.name}`")
                st.markdown(f"**2D Lines:** {len(cfg.lines2d_paths)} files")
                _poly_label = f"`{cfg.expand_polygon_path.name}`" if cfg.expand_polygon_path else f"auto ({cfg.expand_buffer_pct:.0f}%)"
                st.markdown(f"**Expansion:** {_poly_label}")
                st.markdown(f"**Output:** `{cfg.out_cube_path}`")
            with c2:
                st.markdown(f"**Calibration:** `{cfg.calibration.method.value}`")
                st.markdown(f"**Interpolation:** `{cfg.interpolation.method.value}`")
                st.markdown(f"**Backend:** `{cfg.compute.backend.value}`")
                st.markdown(f"**Blend:** {'On' if cfg.blend.enabled else 'Off'} (taper={cfg.blend.taper_width_traces})")


# ════════════════════════════════════════════════════════════════════════
# TAB: Data Explorer
# ════════════════════════════════════════════════════════════════════════

with tab_data:
    section_header("Data Explorer", "📁")

    if st.session_state.meta_3d is None:
        st.info("Run the pipeline first to explore data.")
    else:
        meta = st.session_state.meta_3d

        # 3D metadata
        st.markdown("### 3D Cube Metadata")
        col1, col2, col3 = st.columns(3)
        col1.metric("Traces", f"{meta['n_traces']:,}")
        col2.metric("Samples/Trace", meta["n_samples"])
        col3.metric("Sample Interval", f"{meta['dt_ms']} ms")

        # 2D lines table
        if st.session_state.lines_info:
            st.markdown("### 2D Line Summary")
            import pandas as pd
            df = pd.DataFrame(st.session_state.lines_info)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Map view
        if hasattr(st.session_state, "coords_3d") and st.session_state.coords_3d is not None:
            st.markdown("### Survey Map")

            poly_coords = None
            try:
                cfg = st.session_state.config_obj
                if cfg:
                    from seis2cube.geometry.overlap_detector import OverlapDetector
                    poly = OverlapDetector.load_polygon(cfg.expand_polygon_path)
                    poly_coords = np.array(poly.exterior.coords)
            except Exception:
                pass

            fig = plot_map_with_lines(
                st.session_state.coords_3d,
                getattr(st.session_state, "lines_coords", []),
                getattr(st.session_state, "line_names", []),
                polygon_xy=poly_coords,
            )
            st.plotly_chart(fig, width='stretch')


# ════════════════════════════════════════════════════════════════════════
# TAB: Calibration
# ════════════════════════════════════════════════════════════════════════

with tab_cal:
    section_header("Calibration Results", "🔧")

    before = st.session_state.cal_metrics_before
    after = st.session_state.cal_metrics_after

    if before is None or after is None:
        st.info("Run the pipeline to see calibration results.")
    else:
        # ── What is Calibration? ────────────────────────────────────────
        st.markdown("### What is Calibration?")
        st.info("""
        **Calibration** matches 2D seismic amplitudes to 3D reference data in the overlap zone.

        2D profiles often have different amplitude scaling, time shifts, and phase compared to 3D cubes.
        The calibrator learns these differences from **training pairs** (overlap zone data) and corrects
        all 2D traces. Quality is measured on **test pairs** (held-out data not used for training).
        """)

        # Split info with visual indicator
        n_train = getattr(st.session_state, "n_train", "?")
        n_test = getattr(st.session_state, "n_test", "?")

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Training Pairs", n_train, help="Used to fit the calibration model")
        with col_info2:
            st.metric("Test Pairs (holdout)", n_test, help="Used to evaluate calibration quality (unseen during training)")
        with col_info3:
            st.metric("Holdout Strategy", "Spatial split", help="Last segment of overlap zone held out as test set")

        st.caption("All metrics and plots below are computed on the **test set** (unseen during calibration fitting)")
        st.markdown("<br>", unsafe_allow_html=True)

        # Before / after metrics
        st.markdown("### Before vs After Calibration")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            delta = after["pearson_corr"] - before["corr"]
            render_metric_card(
                "Correlation (before)",
                f"{before['corr']:.4f}",
                icon="🔴",
            )
        with c2:
            render_metric_card(
                "Correlation (after)",
                f"{after['pearson_corr']:.4f}",
                delta=f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}",
                icon="🟢",
            )
        with c3:
            render_metric_card(
                "RMSE (before)",
                f"{before['rmse']:.4f}",
                icon="🔴",
            )
        with c4:
            rmse_delta = before["rmse"] - after["rmse"]
            render_metric_card(
                "RMSE (after)",
                f"{after['rmse']:.4f}",
                delta=f"+{rmse_delta:.4f}" if rmse_delta >= 0 else f"{rmse_delta:.4f}",
                icon="🟢",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Statistical overview: per-pair correlation scatter ────────
        per_before = getattr(st.session_state, "per_corr_before", None)
        per_after = getattr(st.session_state, "per_corr_after", None)
        if per_before is not None and per_after is not None:
            import plotly.graph_objects as go

            st.markdown("### Per-Pair Correlation (test set)")
            st.caption(
                "Each dot = one test pair. X = correlation before calibration, "
                "Y = correlation after. Points above the diagonal improved."
            )

            fig_scatter = go.Figure()
            # Diagonal reference
            fig_scatter.add_trace(go.Scatter(
                x=[-1, 1], y=[-1, 1], mode="lines",
                line=dict(color="grey", dash="dash", width=1),
                showlegend=False, hoverinfo="skip",
            ))
            # Colour by improvement
            improved = per_after > per_before
            colors = np.where(improved, PALETTE["primary"], "#f38ba8")

            fig_scatter.add_trace(go.Scatter(
                x=per_before, y=per_after, mode="markers",
                marker=dict(color=colors, size=6, opacity=0.7,
                            line=dict(width=0.5, color="white")),
                text=[f"Pair #{i+1}<br>Before: {b:.3f}<br>After: {a:.3f}"
                      for i, (b, a) in enumerate(zip(per_before, per_after))],
                hoverinfo="text",
                showlegend=False,
            ))

            n_improved = int(improved.sum())
            fig_scatter.update_layout(
                xaxis_title="Correlation BEFORE",
                yaxis_title="Correlation AFTER",
                xaxis=dict(range=[-1.05, 1.05]),
                yaxis=dict(range=[-1.05, 1.05]),
                height=420, template="plotly_dark",
                paper_bgcolor=PALETTE["bg_card"],
                plot_bgcolor=PALETTE["bg_dark"],
                annotations=[dict(
                    text=f"Improved: {n_improved}/{len(per_before)} pairs",
                    xref="paper", yref="paper", x=0.02, y=0.98,
                    showarrow=False, font=dict(size=13, color=PALETTE["text"]),
                )],
            )
            st.plotly_chart(fig_scatter, width='stretch')

            # Strip plot: distribution of correlations
            st.markdown("### Correlation Distribution")
            fig_strip = go.Figure()
            fig_strip.add_trace(go.Box(
                y=per_before, name="Before", boxpoints="all",
                jitter=0.4, pointpos=-1.5,
                marker=dict(color="#f38ba8", size=4, opacity=0.6),
                line=dict(color="#f38ba8"),
            ))
            fig_strip.add_trace(go.Box(
                y=per_after, name="After", boxpoints="all",
                jitter=0.4, pointpos=-1.5,
                marker=dict(color=PALETTE["primary"], size=4, opacity=0.6),
                line=dict(color=PALETTE["primary"]),
            ))
            fig_strip.update_layout(
                yaxis_title="Pearson Correlation",
                height=350, template="plotly_dark",
                paper_bgcolor=PALETTE["bg_card"],
                plot_bgcolor=PALETTE["bg_dark"],
                showlegend=False,
            )
            st.plotly_chart(fig_strip, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Individual trace browser ─────────────────────────────────
        if hasattr(st.session_state, "sample_traces_2d"):
            st.markdown("### Trace Browser (test set)")
            traces_2d = st.session_state.sample_traces_2d
            traces_3d = st.session_state.sample_traces_3d
            traces_corr = st.session_state.sample_traces_corr
            dt = st.session_state.dt_ms
            n_show = len(traces_2d)

            st.caption(f"Browse **{n_show}** test pairs. Pick a pair to see waveforms and spectra.")
            trace_idx = st.slider("Select test pair", 0, n_show - 1, 0, key="cal_trace_slider")

            # Show per-pair correlation badge
            if per_before is not None and trace_idx < len(per_before):
                cc_b = per_before[trace_idx]
                cc_a = per_after[trace_idx]
                delta_cc = cc_a - cc_b
                badge_color = "green" if delta_cc >= 0 else "red"
                st.markdown(
                    f"**Pair #{trace_idx+1}** — "
                    f"corr before: `{cc_b:.3f}` → after: `{cc_a:.3f}` "
                    f"(:{badge_color}[{'+'if delta_cc>=0 else ''}{delta_cc:.3f}])"
                )

            delrt = st.session_state.meta_3d.get("delrt_ms", 0.0) if st.session_state.meta_3d else 0.0
            time_ax = delrt + np.arange(len(traces_2d[trace_idx])) * dt

            # ── Amplitude scatter: 2D vs 3D before/after ─────────────────────
            st.markdown("#### Amplitude Alignment: 2D vs 3D")
            st.caption("How well do 2D amplitudes match 3D? Perfect alignment = points on diagonal line")

            col_scat1, col_scat2 = st.columns(2)
            with col_scat1:
                # Before calibration scatter
                fig_scat_before = go.Figure()
                t2d = traces_2d[trace_idx]
                t3d = traces_3d[trace_idx]
                # Subsample for visualization if too many points
                n_points = len(t2d)
                step = max(1, n_points // 500)
                fig_scat_before.add_trace(go.Scatter(
                    x=t3d[::step], y=t2d[::step],
                    mode='markers',
                    marker=dict(size=3, color="#f38ba8", opacity=0.6),
                    name='Before calibration'
                ))
                # Add diagonal reference line
                min_val = min(t2d.min(), t3d.min())
                max_val = max(t2d.max(), t3d.max())
                fig_scat_before.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(color='grey', dash='dash', width=1),
                    name='Perfect match', showlegend=False
                ))
                fig_scat_before.update_layout(
                    title="BEFORE Calibration",
                    xaxis_title="3D Amplitude (reference)",
                    yaxis_title="2D Amplitude",
                    height=350,
                    template="plotly_dark",
                    paper_bgcolor=PALETTE["bg_card"],
                    plot_bgcolor=PALETTE["bg_dark"],
                    font=dict(color=PALETTE["text"]),
                    showlegend=False,
                )
                st.plotly_chart(fig_scat_before, width='stretch')

            with col_scat2:
                # After calibration scatter
                fig_scat_after = go.Figure()
                tcorr = traces_corr[trace_idx]
                fig_scat_after.add_trace(go.Scatter(
                    x=t3d[::step], y=tcorr[::step],
                    mode='markers',
                    marker=dict(size=3, color=PALETTE["primary"], opacity=0.6),
                    name='After calibration'
                ))
                fig_scat_after.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(color='grey', dash='dash', width=1),
                    name='Perfect match', showlegend=False
                ))
                fig_scat_after.update_layout(
                    title="AFTER Calibration",
                    xaxis_title="3D Amplitude (reference)",
                    yaxis_title="Calibrated 2D Amplitude",
                    height=350,
                    template="plotly_dark",
                    paper_bgcolor=PALETTE["bg_card"],
                    plot_bgcolor=PALETTE["bg_dark"],
                    font=dict(color=PALETTE["text"]),
                    showlegend=False,
                )
                st.plotly_chart(fig_scat_after, width='stretch')

            # ── Waveform comparison ───────────────────────────────────────
            st.markdown("#### Waveform Comparison")
            fig = plot_trace_comparison(
                traces_2d[trace_idx], traces_3d[trace_idx], traces_corr[trace_idx],
                time_axis=time_ax,
                title=f"Test Pair #{trace_idx + 1} — 2D vs 3D vs Corrected",
            )
            st.plotly_chart(fig, width='stretch')

            # Spectrum comparison
            from seis2cube.utils.spectral import amplitude_spectrum
            freqs, sp2d = amplitude_spectrum(traces_2d[trace_idx], dt)
            _, sp3d = amplitude_spectrum(traces_3d[trace_idx], dt)
            _, spcorr = amplitude_spectrum(traces_corr[trace_idx], dt)

            fig_sp = plot_spectrum_comparison(freqs, sp2d, sp3d, spcorr,
                                             title=f"Spectrum — Test Pair #{trace_idx + 1}")
            st.plotly_chart(fig_sp, width='stretch')


# ════════════════════════════════════════════════════════════════════════
# TAB: Interpolation
# ════════════════════════════════════════════════════════════════════════

with tab_interp:
    section_header("Interpolation Results", "🧩")

    sim = st.session_state.interp_sim_metrics
    cfg_interp = st.session_state.config_obj.interpolation if st.session_state.config_obj else None

    if sim is None:
        st.info("Run the pipeline to see interpolation results.")
    else:
        import plotly.graph_objects as go

        # ── Method Explanation ─────────────────────────────────────────────
        st.markdown("### What is Interpolation?")
        st.info("""
        **Interpolation** fills the gaps between 2D seismic profiles to create a complete 3D volume.

        The algorithm estimates seismic amplitudes at positions where no data was acquired,
        using the calibrated 2D traces as known reference points. The quality of this reconstruction
        is validated by hiding known 3D data and measuring how well the interpolator recovers it.
        """)

        # ── Method Description ────────────────────────────────────────────
        st.markdown("### Selected Interpolation Method")

        method_descriptions = {
            "idw": {
                "name": "Inverse Distance Weighting (IDW)",
                "icon": "📐",
                "description": """
                **How it works:** Each interpolated point is a weighted average of nearby known traces.
                Closer traces have more influence. Fast and simple, but may produce smoother results.
                **Best for:** Quick reconstructions, uniformly spaced 2D lines.
                """,
                "params": ["Power (decay rate)", "Max neighbors"],
            },
            "pocs": {
                "name": "POCS / FPOCS (Projections onto Convex Sets)",
                "icon": "🔄",
                "description": """
                **How it works:** Iterative algorithm that enforces data consistency in both
                spatial and frequency domains. Uses FFT/wavelet transforms and thresholding.
                **Best for:** Complex geology with predictable spectral characteristics.
                """,
                "params": ["Iterations", "Transform (FFT/Wavelet)", "Fast mode"],
            },
            "mssa": {
                "name": "MSSA (Multichannel Singular Spectrum Analysis)",
                "icon": "📊",
                "description": """
                **How it works:** Exploits low-rank structure in seismic data via Hankel matrix decomposition.
                Captures dominant signal patterns and suppresses noise during reconstruction.
                **Best for:** Data with strong coherent signal patterns, noisy acquisitions.
                """,
                "params": ["Rank (signal components)", "Window length"],
            },
        }

        method_key = cfg_interp.method.value if cfg_interp else "idw"
        method_info = method_descriptions.get(method_key, method_descriptions["idw"])

        # Method card with clean vertical layout
        with st.expander(f"{method_info['icon']} {method_info['name']}", expanded=True):
            st.markdown(f"**Method code:** `{method_key}`")
            st.markdown(method_info["description"])
            if cfg_interp:
                st.markdown("---")
                st.markdown("**Configuration Parameters:**")
                if method_key == "idw":
                    st.markdown(f"- IDW Power: `{cfg_interp.idw_power}` (decay rate)")
                    st.markdown(f"- Max Neighbors: `{cfg_interp.idw_max_neighbours}` traces")
                elif method_key == "pocs":
                    st.markdown(f"- Iterations: `{cfg_interp.pocs_niter}`")
                    st.markdown(f"- Transform: `{cfg_interp.pocs_transform.value}`")
                    st.markdown(f"- Fast Mode: `{cfg_interp.pocs_fast}`")
                    st.markdown(f"- Threshold Schedule: `{cfg_interp.pocs_threshold_schedule}`")
                elif method_key == "mssa":
                    st.markdown(f"- Rank: `{cfg_interp.mssa_rank}` (signal components)")
                    st.markdown(f"- Window: `{cfg_interp.mssa_window}` samples")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Grid Overview ────────────────────────────────────────────────
        grid = st.session_state.target_grid
        orig_mask = st.session_state.orig_mask
        sparse_mask = getattr(st.session_state, "sparse_mask", None)

        if grid is not None:
            st.markdown("### Extended Grid Coverage")
            st.caption("Distribution of data sources in the extended 3D volume")

            n_total = grid.n_il * grid.n_xl
            n_orig = int(orig_mask.sum()) if orig_mask is not None else 0
            n_ext_2d = int((sparse_mask & ~orig_mask).sum()) if sparse_mask is not None and orig_mask is not None else 0
            n_empty = n_total - n_orig - n_ext_2d
            pct_orig = 100 * n_orig / max(n_total, 1)
            pct_2d = 100 * n_ext_2d / max(n_total, 1)
            pct_empty = 100 * n_empty / max(n_total, 1)

            # Visual progress bar showing data distribution
            st.markdown("#### Data Source Distribution")
            # Ensure minimum width of 1 for each column to prevent st.columns error
            col_widths = [max(1, int(pct_orig)), max(1, int(pct_2d)), max(1, int(pct_empty))]
            bar_col1, bar_col2, bar_col3 = st.columns(col_widths)
            with bar_col1:
                st.markdown(f"<div style='background:{PALETTE['primary']};height:20px;border-radius:4px;'></div>",
                           unsafe_allow_html=True)
            with bar_col2:
                st.markdown(f"<div style='background:#a6e3a1;height:20px;border-radius:4px;'></div>",
                           unsafe_allow_html=True)
            with bar_col3:
                st.markdown(f"<div style='background:#45475a;height:20px;border-radius:4px;'></div>",
                           unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Total Grid", f"{grid.n_il} × {grid.n_xl}", icon="🗂️")
            with c2:
                render_metric_card("3D Original", f"{n_orig:,} ({pct_orig:.1f}%)", icon="🔷")
            with c3:
                render_metric_card("2D Calibrated", f"{n_ext_2d:,} ({pct_2d:.1f}%)", icon="📏")
            with c4:
                render_metric_card("Interpolated", f"{n_empty:,} ({pct_empty:.1f}%)", icon="✨")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Simulation Quality Metrics ────────────────────────────────────
        st.markdown("### Reconstruction Quality (Simulation)")
        st.info("""
        **How we measure quality:** Inside the original 3D area, we hide some traces and let the
        interpolator reconstruct them. Then we compare the reconstructed traces with the actual data.
        This "simulation" shows how well the method works before applying it to the unknown extension area.
        """)

        # Quality interpretation
        corr_val = sim.get("pearson_corr", 0)
        rmse_val = sim.get("rmse", 0)

        if corr_val > 0.8:
            quality_level = ("🟢 Excellent", "Correlation > 0.8: High-quality reconstruction expected")
        elif corr_val > 0.5:
            quality_level = ("🟡 Good", "Correlation 0.5-0.8: Acceptable quality with some artifacts")
        else:
            quality_level = ("🔴 Fair", "Correlation < 0.5: May need more 2D profiles or different method")

        st.markdown(f"**Overall Quality:** {quality_level[0]} — *{quality_level[1]}*")

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card(
                "RMSE",
                f"{rmse_val:.2f}",
                icon="📐",
                delta="lower is better",
            )
        with c2:
            render_metric_card(
                "MAE",
                f"{sim.get('mae', 0):.2f}",
                icon="📏",
                delta="mean absolute error",
            )
        with c3:
            render_metric_card(
                "Correlation",
                f"{corr_val:.3f}",
                icon="🎯",
                delta=f"quality: {quality_level[0].split()[1]}",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Coverage Visualization ────────────────────────────────────────
        if orig_mask is not None:
            st.markdown("### Data Coverage Map")
            st.caption("Visual representation of data sources: Blue = 3D, Green = 2D, Dark = Interpolated")

            # Build 3-value map: 0=empty, 1=2D-only, 2=original-3D
            coverage = np.zeros_like(orig_mask, dtype=np.float32)
            if sparse_mask is not None:
                ext_2d = sparse_mask & ~orig_mask
                coverage[ext_2d] = 1.0
            coverage[orig_mask] = 2.0

            # Get actual IL/XL labels for hover
            il_labels = grid.inlines if grid else np.arange(coverage.shape[0])
            xl_labels = grid.xlines if grid else np.arange(coverage.shape[1])

            fig_cov = go.Figure(data=go.Heatmap(
                z=coverage,
                x=xl_labels,
                y=il_labels,
                colorscale=[
                    [0.0, "#313244"],  # Empty: dark gray
                    [0.5, "#a6e3a1"],  # 2D profiles: light green
                    [1.0, PALETTE["primary"]],  # 3D: indigo
                ],
                zmin=0, zmax=2,
                showscale=False,
                hovertemplate=(
                    "<b>Inline:</b> %{y}<br>"
                    "<b>Crossline:</b> %{x}<br>"
                    "<b>Source:</b> " +
                    "%{z:,.0f}<extra></extra>"
                ),
            ))

            # Add custom hover labels
            hover_text = []
            for i in range(coverage.shape[0]):
                row = []
                for j in range(coverage.shape[1]):
                    val = coverage[i, j]
                    source = "Original 3D" if val == 2 else ("2D Profile" if val == 1 else "Interpolated")
                    row.append(f"IL: {il_labels[i]}<br>XL: {xl_labels[j]}<br>Source: {source}")
                hover_text.append(row)

            fig_cov.update_traces(text=hover_text, hovertemplate="%{text}<extra></extra>")
            fig_cov.update_layout(
                height=500,
                paper_bgcolor=PALETTE["bg_dark"],
                plot_bgcolor=PALETTE["bg_dark"],
                font=dict(color=PALETTE["text"]),
                xaxis_title="Crossline",
                yaxis_title="Inline",
                yaxis=dict(autorange="reversed", scaleanchor="x"),
                xaxis=dict(constrain="domain"),
                margin=dict(l=70, r=20, t=30, b=50),
            )
            st.plotly_chart(fig_cov, width='stretch')

            # Legend
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<span style='color:{PALETTE['primary']}'>🔷</span> **3D Original** ({pct_orig:.1f}%)",
                           unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color:#a6e3a1'>📏</span> **2D Profiles** ({pct_2d:.1f}%)",
                           unsafe_allow_html=True)
            with col3:
                st.markdown(f"<span style='color:#45475a'>✨</span> **Interpolated** ({pct_empty:.1f}%)",
                           unsafe_allow_html=True)
            with col4:
                st.markdown(f"📊 **Total Traces:** {n_total:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Scatter Plot: True vs Interpolated ───────────────────────────
        st.markdown("### True vs Interpolated Amplitudes")
        st.caption("How well does the interpolation reconstruct hidden 3D data? Points on diagonal = perfect reconstruction")

        scatter_true = getattr(st.session_state, "interp_scatter_true", None)
        scatter_pred = getattr(st.session_state, "interp_scatter_pred", None)

        if scatter_true is not None and scatter_pred is not None and len(scatter_true) > 0:
            fig_interp_scatter = go.Figure()

            # Main scatter
            fig_interp_scatter.add_trace(go.Scatter(
                x=scatter_true, y=scatter_pred,
                mode='markers',
                marker=dict(
                    size=5, color=PALETTE["primary"], opacity=0.5,
                    line=dict(width=0.5, color="white")
                ),
                name='Interpolated vs True',
            ))

            # Diagonal reference line (perfect match)
            min_val = min(scatter_true.min(), scatter_pred.min())
            max_val = max(scatter_true.max(), scatter_pred.max())
            fig_interp_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color='#f38ba8', dash='dash', width=2),
                name='Perfect match (y=x)',
            ))

            # ±20% error band
            fig_interp_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val * 1.2, max_val * 1.2],
                mode='lines', line=dict(color='grey', dash='dot', width=1),
                name='+20% error',
                showlegend=False,
            ))
            fig_interp_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val * 0.8, max_val * 0.8],
                mode='lines', line=dict(color='grey', dash='dot', width=1),
                name='-20% error',
                showlegend=False,
            ))

            fig_interp_scatter.update_layout(
                title=f"Interpolation Quality (n={len(scatter_true)} traces)",
                xaxis_title="True 3D Amplitude (RMS)",
                yaxis_title="Interpolated Amplitude (RMS)",
                height=450,
                template="plotly_dark",
                paper_bgcolor=PALETTE["bg_card"],
                plot_bgcolor=PALETTE["bg_dark"],
                font=dict(color=PALETTE["text"]),
                xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal axes
                yaxis=dict(scaleanchor="x", scaleratio=1),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_interp_scatter, width='stretch')

            # Statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                mae_amp = np.mean(np.abs(scatter_pred - scatter_true))
                st.metric("Mean Abs Error (amp)", f"{mae_amp:.3f}")
            with col_stat2:
                bias = np.mean(scatter_pred - scatter_true)
                st.metric("Bias (pred - true)", f"{bias:.3f}",
                         delta="overestimates" if bias > 0 else "underestimates")
            with col_stat3:
                within_20pct = 100 * np.mean(
                    (scatter_pred >= 0.8 * scatter_true) & (scatter_pred <= 1.2 * scatter_true)
                )
                st.metric("Within ±20%", f"{within_20pct:.1f}%")
        else:
            st.info("Scatter plot data not available. Run pipeline to generate comparison.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Recommendations ───────────────────────────────────────────────
        with st.expander("💡 Recommendations for Better Results", expanded=False):
            st.markdown("""
            **To improve interpolation quality:**

            1. **Add more 2D profiles** — More calibrated traces = better spatial sampling
            2. **Space profiles evenly** — Uniform coverage reduces artifacts
            3. **Extend overlap zone** — Larger calibration area = more accurate matching
            4. **Try different methods:**
               - Use **POCS** for complex geology with predictable patterns
               - Use **MSSA** for noisy data with coherent signal
               - Use **IDW** for quick, robust reconstructions
            5. **Adjust buffer size** — Smaller expansion needs less interpolation
            """)


# ════════════════════════════════════════════════════════════════════════
# TAB: Volume Viewer
# ════════════════════════════════════════════════════════════════════════

with tab_viewer:
    section_header("Volume Viewer", "🗺️")

    vol = st.session_state.final_volume
    orig_vol = st.session_state.cube_volume
    grid = st.session_state.target_grid

    if vol is None:
        st.info("Run the pipeline to view volumes.")
    else:
        # ── Survey Map with 2D lines and 3D boundary ──────────────────────
        st.markdown("### Survey Map with 2D Profiles & 3D Boundary")
        st.caption("Interactive map showing original 3D area (blue), expansion zone (orange dashed), and 2D profiles")

        poly_coords = None
        try:
            cfg = st.session_state.config_obj
            if cfg and cfg.expand_polygon_path:
                from seis2cube.geometry.overlap_detector import OverlapDetector
                poly = OverlapDetector.load_polygon(cfg.expand_polygon_path)
                poly_coords = np.array(poly.exterior.coords)
            elif hasattr(st.session_state, 'expand_poly'):
                poly_coords = st.session_state.expand_poly
        except Exception:
            pass

        # Create enhanced map with 2D traces, 3D boundary, and expansion polygon
        fig_map = plot_map_with_lines(
            st.session_state.coords_3d,
            getattr(st.session_state, "lines_coords", []),
            getattr(st.session_state, "line_names", []),
            polygon_xy=poly_coords,
            show_3d_boundary=True,
            inlines_3d=st.session_state.inlines_3d,
            xlines_3d=st.session_state.xlines_3d,
            grid=grid,
            show_extended_grid=True,
            title="Survey Geometry: 3D Area + 2D Profiles + Expansion Zone",
            height=500,
        )
        st.plotly_chart(fig_map, width='stretch')

        # Map legend explanation
        st.markdown("""
        **Map Legend:**
        - 🔷 **Blue solid** = Original 3D cube area (source of calibration reference)
        - ⬜ **Gray dashed** = Extended grid area (target for reconstruction)
        - 📏 **Colored lines** = 2D seismic profiles (input data for extension)
        - 🟠 **Orange dotted** = Expansion polygon (optional custom boundary)

        **Axes:** Equal scale (1:1) — 1 meter on X = 1 meter on Y
        """)

        st.divider()

        # ── Section Viewers ───────────────────────────────────────────────
        st.markdown("### Seismic Sections")

        view_mode = st.radio(
            "View Mode",
            ["Time Slice (IL × XL)", "Inline Section (XL × Time)", "Crossline Section (IL × Time)"],
            horizontal=True,
            key="viewer_mode",
            help="Time Slice = map view at fixed time. Inline/Crossline = vertical sections through the cube",
        )

        volume_sel = st.radio(
            "Volume to Display",
            ["Final (extended)", "Original 3D", "Side by Side Comparison"],
            horizontal=True,
            key="viewer_vol",
        )

        _delrt = st.session_state.meta_3d.get("delrt_ms", 0.0) if st.session_state.meta_3d else 0.0

        # Prepare polygon overlays for time slice maps (convert XY to IL/XL)
        lines_ilxl = None
        try:
            geom = st.session_state.geom
            lines_coords = getattr(st.session_state, "lines_coords", [])
            if geom is not None and lines_coords:
                lines_ilxl = []
                for lc in lines_coords:
                    if len(lc) > 0:
                        il_frac, xl_frac = geom.xy_to_ilxl(lc[:, 0], lc[:, 1])
                        lines_ilxl.append(np.column_stack([il_frac, xl_frac]))
        except Exception:
            lines_ilxl = None

        if view_mode == "Time Slice (IL × XL)":
            max_samp = vol.shape[2] - 1
            t_idx = st.slider(
                "Time Sample Index",
                0, max_samp, max_samp // 3,
                key="ts_slider",
                help="Select time slice to view (0 = earliest time)",
            )
            t_ms = _delrt + t_idx * (grid.dt_ms if grid else 2.0)
            st.caption(f"**Time: {t_ms:.1f} ms** | Sample {t_idx} of {max_samp}")

            if volume_sel == "Side by Side Comparison" and orig_vol is not None:
                c1, c2 = st.columns(2)
                with c1:
                    fig_o = plot_time_slice(
                        orig_vol, t_idx,
                        inlines=st.session_state.inlines_3d,
                        xlines=st.session_state.xlines_3d,
                        title=f"Original 3D — t={t_ms:.1f} ms",
                        height=450,
                        orig_inlines=st.session_state.inlines_3d,
                        orig_xlines=st.session_state.xlines_3d,
                        extended_inlines=grid.inlines if grid else None,
                        extended_xlines=grid.xlines if grid else None,
                        lines_ilxl=lines_ilxl,
                    )
                    st.plotly_chart(fig_o, width='stretch')
                with c2:
                    fig_f = plot_time_slice(
                        vol, t_idx,
                        inlines=grid.inlines if grid else None,
                        xlines=grid.xlines if grid else None,
                        title=f"Extended Cube — t={t_ms:.1f} ms",
                        height=450,
                        orig_inlines=st.session_state.inlines_3d,
                        orig_xlines=st.session_state.xlines_3d,
                        extended_inlines=grid.inlines if grid else None,
                        extended_xlines=grid.xlines if grid else None,
                        lines_ilxl=lines_ilxl,
                    )
                    st.plotly_chart(fig_f, width='stretch')
            else:
                show_vol = vol if "Final" in volume_sel else (orig_vol if orig_vol is not None else vol)
                label = "Extended" if "Final" in volume_sel else "Original"
                fig = plot_time_slice(
                    show_vol, t_idx,
                    inlines=grid.inlines if grid else None,
                    xlines=grid.xlines if grid else None,
                    title=f"{label} — Time Slice at t={t_ms:.1f} ms",
                    height=500,
                    orig_inlines=st.session_state.inlines_3d,
                    orig_xlines=st.session_state.xlines_3d,
                    extended_inlines=grid.inlines if grid else None,
                    extended_xlines=grid.xlines if grid else None,
                    lines_ilxl=lines_ilxl,
                )
                st.plotly_chart(fig, width='stretch')

        elif view_mode == "Inline Section (XL × Time)":
            max_il = vol.shape[0] - 1
            il_idx = st.slider(
                "Inline Index",
                0, max_il, max_il // 2,
                key="il_slider",
                help="Select inline number (vertical section perpendicular to inline direction)",
            )
            inline_label = grid.inlines[il_idx] if grid is not None and il_idx < len(grid.inlines) else il_idx
            st.caption(f"**Inline: {inline_label}** | Index {il_idx} of {max_il}")

            time_ax = _delrt + np.arange(vol.shape[2]) * (grid.dt_ms if grid else 2.0)

            if volume_sel == "Side by Side Comparison" and orig_vol is not None:
                c1, c2 = st.columns(2)
                with c1:
                    il_o = min(il_idx, orig_vol.shape[0] - 1)
                    fig = plot_inline_section(
                        orig_vol, il_o,
                        xlines=st.session_state.xlines_3d,
                        time_axis=time_ax[:orig_vol.shape[2]],
                        title=f"Original — Inline {inline_label}",
                        height=450,
                    )
                    st.plotly_chart(fig, width='stretch')
                with c2:
                    fig = plot_inline_section(
                        vol, il_idx,
                        xlines=grid.xlines if grid else None,
                        time_axis=time_ax,
                        title=f"Extended — Inline {inline_label}",
                        height=450,
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                show_vol = vol if "Final" in volume_sel else (orig_vol if orig_vol is not None else vol)
                idx = min(il_idx, show_vol.shape[0] - 1)
                xlines_to_use = grid.xlines if grid is not None else None
                if "Original" in volume_sel and st.session_state.xlines_3d is not None:
                    xlines_to_use = st.session_state.xlines_3d
                fig = plot_inline_section(
                    show_vol, idx,
                    xlines=xlines_to_use,
                    time_axis=time_ax[:show_vol.shape[2]],
                    title=f"{'Extended' if 'Final' in volume_sel else 'Original'} — Inline Section",
                    height=500,
                )
                st.plotly_chart(fig, width='stretch')

        else:  # Crossline Section
            max_xl = vol.shape[1] - 1
            xl_idx = st.slider(
                "Crossline Index",
                0, max_xl, max_xl // 2,
                key="xl_slider",
                help="Select crossline number (vertical section perpendicular to crossline direction)",
            )
            xline_label = grid.xlines[xl_idx] if grid is not None and xl_idx < len(grid.xlines) else xl_idx
            st.caption(f"**Crossline: {xline_label}** | Index {xl_idx} of {max_xl}")

            time_ax = _delrt + np.arange(vol.shape[2]) * (grid.dt_ms if grid else 2.0)

            if volume_sel == "Side by Side Comparison" and orig_vol is not None:
                c1, c2 = st.columns(2)
                with c1:
                    xl_o = min(xl_idx, orig_vol.shape[1] - 1)
                    fig = plot_crossline_section(
                        orig_vol, xl_o,
                        inlines=st.session_state.inlines_3d,
                        time_axis=time_ax[:orig_vol.shape[2]],
                        title=f"Original — Crossline {xline_label}",
                        height=450,
                    )
                    st.plotly_chart(fig, width='stretch')
                with c2:
                    fig = plot_crossline_section(
                        vol, xl_idx,
                        inlines=grid.inlines if grid else None,
                        time_axis=time_ax,
                        title=f"Extended — Crossline {xline_label}",
                        height=450,
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                show_vol = vol if "Final" in volume_sel else (orig_vol if orig_vol is not None else vol)
                idx = min(xl_idx, show_vol.shape[1] - 1)
                inlines_to_use = grid.inlines if grid is not None else None
                if "Original" in volume_sel and st.session_state.inlines_3d is not None:
                    inlines_to_use = st.session_state.inlines_3d
                fig = plot_crossline_section(
                    show_vol, idx,
                    inlines=inlines_to_use,
                    time_axis=time_ax[:show_vol.shape[2]],
                    title=f"{'Extended' if 'Final' in volume_sel else 'Original'} — Crossline Section",
                    height=500,
                )
                st.plotly_chart(fig, width='stretch')


# ════════════════════════════════════════════════════════════════════════
# TAB: QC Report
# ════════════════════════════════════════════════════════════════════════

with tab_qc:
    section_header("Quality Control Report", "📈")

    before = st.session_state.cal_metrics_before
    after = st.session_state.cal_metrics_after
    sim = st.session_state.interp_sim_metrics

    if before is None and sim is None:
        st.info("Run the pipeline to generate the QC report.")
    else:
        # Summary table
        rows = []
        if before and after:
            rows.append({"Metric": "Calibration Corr (before)", "Value": f"{before['corr']:.4f}"})
            rows.append({"Metric": "Calibration Corr (after)", "Value": f"{after['pearson_corr']:.4f}"})
            rows.append({"Metric": "Calibration RMSE (before)", "Value": f"{before['rmse']:.4f}"})
            rows.append({"Metric": "Calibration RMSE (after)", "Value": f"{after['rmse']:.4f}"})
            if "mae" in after:
                rows.append({"Metric": "Calibration MAE", "Value": f"{after['mae']:.4f}"})
            if "spectral_l2_rel" in after:
                rows.append({"Metric": "Spectral L2 (rel)", "Value": f"{after['spectral_l2_rel']:.4f}"})
        if sim:
            rows.append({"Metric": "Interpolation RMSE (sim)", "Value": f"{sim['rmse']:.4f}"})
            rows.append({"Metric": "Interpolation MAE (sim)", "Value": f"{sim['mae']:.4f}"})
            rows.append({"Metric": "Interpolation Corr (sim)", "Value": f"{sim['pearson_corr']:.4f}"})

        # Metrics with explanations
        st.markdown("### Metrics Reference Guide")
        st.caption("Click on metric name to see what it means and how to interpret")

        metric_explanations = {
            "Calibration Corr": {
                "description": "Pearson correlation between 2D and 3D amplitudes",
                "interpretation": "+1 = perfect match, 0 = no correlation, -1 = inverse. Above 0.7 is good.",
                "good_range": "> 0.7",
            },
            "Calibration RMSE": {
                "description": "Root Mean Square Error of amplitude differences",
                "interpretation": "Lower is better. Typical seismic amplitudes are 1000-10000. Compare before/after.",
                "good_range": "↓ lower",
            },
            "Calibration MAE": {
                "description": "Mean Absolute Error of amplitude differences",
                "interpretation": "Average absolute difference. Less sensitive to outliers than RMSE.",
                "good_range": "↓ lower",
            },
            "Spectral L2": {
                "description": "Spectral (frequency domain) L2 norm difference",
                "interpretation": "Measures frequency content mismatch. Lower = better spectral matching.",
                "good_range": "↓ < 1.0",
            },
            "Interpolation RMSE": {
                "description": "RMSE of reconstructed vs true 3D amplitudes (simulation)",
                "interpretation": "How well interpolation fills gaps. Tested on hidden 3D data.",
                "good_range": "↓ lower",
            },
            "Interpolation MAE": {
                "description": "MAE of reconstructed vs true 3D amplitudes",
                "interpretation": "Average reconstruction error. Typical: 10-30% of signal amplitude.",
                "good_range": "↓ lower",
            },
            "Interpolation Corr": {
                "description": "Correlation between interpolated and true 3D amplitudes",
                "interpretation": "Quality of gap filling. > 0.5 acceptable, > 0.8 excellent.",
                "good_range": "> 0.5",
            },
        }

        # Display metrics with expanders for explanations
        if before and after:
            with st.expander("🔧 Calibration Metrics", expanded=True):
                col_m1, col_m2, col_m3 = st.columns([2, 1, 3])
                with col_m1:
                    st.markdown("**Metric**")
                with col_m2:
                    st.markdown("**Value**")
                with col_m3:
                    st.markdown("**What it means**")

                # Correlation before
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("Correlation (before)")
                with col2:
                    corr_b = before['corr']
                    color = "🟢" if abs(corr_b) > 0.5 else "🟡" if abs(corr_b) > 0.3 else "🔴"
                    st.markdown(f"{color} `{corr_b:.4f}`")
                with col3:
                    st.caption("Raw 2D vs 3D match before any calibration")

                # Correlation after
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("**Correlation (after)**")
                with col2:
                    corr_a = after['pearson_corr']
                    color = "🟢" if abs(corr_a) > 0.7 else "🟡" if abs(corr_a) > 0.5 else "🔴"
                    delta = f" ({corr_a - corr_b:+.3f})"
                    st.markdown(f"{color} **`{corr_a:.4f}`{delta}**")
                with col3:
                    st.caption("After calibration. Should be higher than 'before'. Target: > 0.7")

                # RMSE before
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("RMSE (before)")
                with col2:
                    st.markdown(f"`{before['rmse']:.1f}`")
                with col3:
                    st.caption("Amplitude error before calibration (typical: 1000-15000)")

                # RMSE after
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("**RMSE (after)**")
                with col2:
                    rmse_a = after['rmse']
                    improvement = (1 - rmse_a / before['rmse']) * 100 if before['rmse'] > 0 else 0
                    st.markdown(f"**`{rmse_a:.1f}`** ({improvement:.0f}% better)")
                with col3:
                    st.caption("Lower is better. Good calibration reduces RMSE by 30-70%")

                # MAE
                if "mae" in after:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        st.markdown("MAE")
                    with col2:
                        st.markdown(f"`{after['mae']:.1f}`")
                    with col3:
                        st.caption("Mean absolute amplitude error. More robust than RMSE.")

                # Spectral
                if "spectral_l2_rel" in after:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        st.markdown("Spectral L2 (rel)")
                    with col2:
                        spec = after['spectral_l2_rel']
                        color = "🟢" if spec < 1.0 else "🟡" if spec < 2.0 else "🔴"
                        st.markdown(f"{color} `{spec:.3f}`")
                    with col3:
                        st.caption("Frequency content match. < 1.0 = good spectral alignment")

        if sim:
            with st.expander("🧩 Interpolation Metrics", expanded=True):
                col_m1, col_m2, col_m3 = st.columns([2, 1, 3])
                with col_m1:
                    st.markdown("**Metric**")
                with col_m2:
                    st.markdown("**Value**")
                with col_m3:
                    st.markdown("**What it means**")

                # Interpolation Corr
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("**Correlation (sim)**")
                with col2:
                    interp_corr = sim['pearson_corr']
                    color = "🟢" if interp_corr > 0.7 else "🟡" if interp_corr > 0.5 else "🔴"
                    st.markdown(f"{color} **`{interp_corr:.4f}`**")
                with col3:
                    st.caption("How well interpolation recovers hidden 3D. > 0.5 OK, > 0.8 excellent")

                # Interpolation RMSE
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("RMSE (sim)")
                with col2:
                    st.markdown(f"`{sim['rmse']:.1f}`")
                with col3:
                    st.caption("Reconstruction error on simulated gaps. Lower = better filling.")

                # Interpolation MAE
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown("MAE (sim)")
                with col2:
                    st.markdown(f"`{sim['mae']:.1f}`")
                with col3:
                    st.caption("Average reconstruction error. Compare to signal amplitude.")

        # Summary interpretation
        st.info("""
        **How to read this report:**

        **Calibration** should improve correlation (higher) and reduce RMSE (lower).
        If 'after' is worse than 'before', the calibrator may be overfitting or the overlap zone is too small.

        **Interpolation** quality depends on 2D line spacing and method choice.
        Correlation > 0.5 is acceptable, > 0.7 is good. IDW is robust but may smooth details.
        POCS/MSSA can capture complex patterns but need sufficient 2D coverage.
        """)

        # Calibration improvement bar chart
        if before and after:
            st.markdown("### Calibration Improvement")
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Bar(name="Before", x=["Correlation", "1 − RMSE"],
                       y=[before["corr"], 1 - before["rmse"]],
                       marker_color=PALETTE["danger"]),
                go.Bar(name="After", x=["Correlation", "1 − RMSE"],
                       y=[after["pearson_corr"], 1 - after["rmse"]],
                       marker_color=PALETTE["success"]),
            ])
            fig.update_layout(
                barmode="group", height=350,
                paper_bgcolor=PALETTE["bg_dark"],
                plot_bgcolor=PALETTE["bg_dark"],
                font=dict(color=PALETTE["text"]),
            )
            st.plotly_chart(fig, width='stretch')

        # Download QC JSON
        cfg = st.session_state.config_obj
        if cfg is not None:
            qc_json_path = cfg.qc_report_dir / "qc_report.json"
            if qc_json_path.exists():
                st.download_button(
                    "📥 Download QC Report (JSON)",
                    data=qc_json_path.read_text(),
                    file_name="qc_report.json",
                    mime="application/json",
                )
