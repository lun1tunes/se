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
    add_log,
    init_state,
    set_stage,
)
from seis2cube.ui.components import (
    PALETTE,
    SEISMIC_COLORSCALE,
    plot_convergence,
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

# ── Global CSS ──────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background-color: {PALETTE['bg_dark']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PALETTE['bg_card']};
        border-right: 1px solid #313244;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {{
        color: {PALETTE['text']};
    }}
    h1, h2, h3 {{
        color: {PALETTE['text']} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {PALETTE['bg_card']};
        border-radius: 8px 8px 0 0;
        border: 1px solid #313244;
        color: {PALETTE['text_muted']};
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PALETTE['primary']};
        color: white !important;
        border-color: {PALETTE['primary']};
    }}
    div[data-testid="stExpander"] {{
        background-color: {PALETTE['bg_card']};
        border: 1px solid #313244;
        border-radius: 12px;
    }}
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] label,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
        color: {PALETTE['text']} !important;
    }}
    div[data-testid="stExpander"] summary > span {{
        padding-left: 4px;
    }}
    div[data-testid="stExpander"] code {{
        color: {PALETTE['primary']} !important;
        background-color: rgba(137, 180, 250, 0.12) !important;
    }}
    .stSelectbox label, .stSlider label, .stNumberInput label, .stFileUploader label,
    .stTextInput label {{
        color: {PALETTE['text']} !important;
    }}
    /* Fix overlapping file-uploader text */
    [data-testid="stFileUploader"] section {{
        padding: 0;
    }}
    [data-testid="stFileUploader"] section > div {{
        padding-top: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding:16px 0;">
                <span style="font-size:2rem;">🌊</span>
                <h2 style="margin:4px 0 0 0; color:{PALETTE['text']};">seis2cube</h2>
                <p style="color:{PALETTE['text_muted']}; font-size:0.8rem; margin:0;">
                    3D SEG-Y Extension from 2D Profiles
                </p>
            </div>
            <hr style="border-color:#313244;">
            """,
            unsafe_allow_html=True,
        )

        # ── Data paths ─────────────────────────────────────────────────
        st.markdown(f"**📂 Input Data**")

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

        # ── Settings ──────────────────────────────────────────────────
        st.divider()
        st.markdown(f"**🎛️ Settings**")

        expand_pct = st.slider(
            "Expansion buffer, % of 3D size",
            min_value=0, max_value=200, value=50, step=5,
            help="Buffer around 3D hull as % of mean side. 50% extends half a cube width on each side (area ≈ 4×).",
        )

        cal_opts = ["global_shift", "windowed", "linear_regression", "gbdt"]
        cal_method = st.selectbox("Calibration method", cal_opts, index=0)

        interp_opts = ["idw", "pocs", "mssa"]
        interp_method = st.selectbox("Interpolation method", interp_opts, index=0)

        blend_traces = st.number_input(
            "Blend taper (traces)", min_value=0, max_value=50, value=10, step=1,
            help="Cosine taper width at the seam between original and extended zones.",
        )

        # ── Build config from UI fields ───────────────────────────────
        if _cube_files and _line_files:
            try:
                from seis2cube.config import (
                    PipelineConfig, CalibrationMethod, InterpolationMethod,
                )
                cfg_obj = PipelineConfig(
                    cube3d_path=_cube_files[0],
                    lines2d_paths=_line_files,
                    expand_polygon_path=None,
                    expand_buffer_pct=float(expand_pct),
                )
                cfg_obj.calibration.method = CalibrationMethod(cal_method)
                cfg_obj.interpolation.method = InterpolationMethod(interp_method)
                cfg_obj.blend.taper_width_traces = int(blend_traces)
                st.session_state.config_obj = cfg_obj
            except Exception as e:
                st.error(f"Config error: {e}")

        # ── Pipeline control ────────────────────────────────────────────
        st.divider()
        st.markdown(f"**🚀 Pipeline**")

        stage = st.session_state.pipeline_stage
        if stage == PipelineStage.IDLE:
            render_status_badge("Ready", "info")
        elif stage == PipelineStage.DONE:
            render_status_badge("Complete", "success")
        elif stage == PipelineStage.ERROR:
            render_status_badge("Error", "error")
        else:
            render_status_badge("Running", "running")

        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button(
                "▶ Run",
                use_container_width=True,
                disabled=st.session_state.config_obj is None or stage not in (
                    PipelineStage.IDLE, PipelineStage.DONE, PipelineStage.ERROR
                ),
                type="primary",
            )
        with col2:
            reset_btn = st.button("↺ Reset", use_container_width=True)

        if reset_btn:
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if run_btn:
            _run_pipeline()

        # ── Log ─────────────────────────────────────────────────────────
        if st.session_state.pipeline_log:
            st.divider()
            with st.expander("📋 Pipeline Log", expanded=False):
                for msg in st.session_state.pipeline_log[-30:]:
                    st.text(msg)


# ── Pipeline runner (with UI progress updates) ─────────────────────────


def _run_pipeline() -> None:
    """Execute the pipeline with progress feedback into session state."""
    cfg = st.session_state.config_obj
    if cfg is None:
        st.error("No config loaded")
        return

    st.session_state.pipeline_log = []
    total_steps = len(STAGE_ORDER)
    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()

    try:
        from seis2cube.io.segy_dataset import SegyDataset
        from seis2cube.geometry.crs_converter import CRSConverter
        from seis2cube.geometry.geometry_model import AffineGridMapper, KDTreeMapper
        from seis2cube.geometry.overlap_detector import OverlapDetector
        from seis2cube.pipeline.runner import (
            PipelineRunner,
            _make_calibrator,
            _make_interpolator,
        )
        from seis2cube.pipeline.volume_builder import VolumeBuilder
        from seis2cube.io.segy_writer import SegyWriter3D
        from seis2cube.calibration.base import CalibrationPair
        from seis2cube.qc.reporter import QCReporter

        def _progress(idx: int, stage: PipelineStage):
            p = (idx + 1) / total_steps
            set_stage(stage, p)
            progress_bar.progress(p)
            status_text.markdown(f"**{STAGE_LABELS[stage]}**")
            add_log(STAGE_LABELS[stage])

        runner = PipelineRunner(cfg)

        # 1. Ingest 3D
        _progress(0, PipelineStage.INGESTING)
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

        # 2. CRS
        _progress(1, PipelineStage.CRS_CONVERSION)
        crs_conv = CRSConverter(cfg.crs)
        if not crs_conv.is_identity:
            cx, cy = crs_conv.forward(coords_3d[:, 0], coords_3d[:, 1])
            coords_3d = np.column_stack([cx, cy])

        # 3. Geometry
        _progress(2, PipelineStage.GEOMETRY)
        if inlines_3d is not None and xlines_3d is not None:
            geom = AffineGridMapper(coords_3d, inlines_3d, xlines_3d)
        else:
            ilxl = np.zeros((len(coords_3d), 2))
            geom = KDTreeMapper(coords_3d, ilxl[:, 0], ilxl[:, 1])

        # 4. Overlap
        _progress(3, PipelineStage.OVERLAP)
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

        # 5. Load 2D
        _progress(4, PipelineStage.LOADING_2D)
        lines_2d = []
        lines_info = []
        for lpath in cfg.lines2d_paths:
            line = runner._load_2d_line(lpath, crs_conv, meta3d.dt_ms, meta3d.n_samples,
                                             target_delrt_ms=meta3d.delrt_ms)
            lines_2d.append(line)
            lines_info.append({"name": line.name, "n_traces": line.n_traces, "dt_ms": line.dt_ms})
        st.session_state.lines_info = lines_info
        st.session_state.lines_coords = [l.coords for l in lines_2d]
        st.session_state.line_names = [l.name for l in lines_2d]

        # 6. Calibration
        _progress(5, PipelineStage.CALIBRATION)
        train_pairs, test_pairs = runner._build_calibration_pairs(
            lines_2d, overlap, geom, meta3d, cube_volume, inlines_3d, xlines_3d,
        )
        calibrator = _make_calibrator(cfg)
        cal_model = calibrator.fit(train_pairs)

        # Evaluate
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

            # Store ALL test traces for visualisation
            n_test = len(test_pairs.amp_2d)
            st.session_state.sample_traces_2d = test_pairs.amp_2d
            st.session_state.sample_traces_3d = test_pairs.amp_3d
            corrected_test = calibrator._apply_array(test_pairs.amp_2d, cal_model)
            st.session_state.sample_traces_corr = corrected_test
            st.session_state.dt_ms = meta3d.dt_ms
            st.session_state.n_train = len(train_pairs.amp_2d)
            st.session_state.n_test = n_test

            # Per-pair correlation stats (before & after calibration)
            per_corr_before = []
            per_corr_after = []
            for i in range(n_test):
                cc_b = float(np.corrcoef(test_pairs.amp_2d[i], test_pairs.amp_3d[i])[0, 1])
                cc_a = float(np.corrcoef(corrected_test[i], test_pairs.amp_3d[i])[0, 1])
                per_corr_before.append(cc_b if np.isfinite(cc_b) else 0.0)
                per_corr_after.append(cc_a if np.isfinite(cc_a) else 0.0)
            st.session_state.per_corr_before = np.array(per_corr_before)
            st.session_state.per_corr_after = np.array(per_corr_after)

        # 7. Apply calibration
        _progress(6, PipelineStage.APPLYING_CAL)
        calibrated_lines = [calibrator.apply(line, cal_model) for line in lines_2d]

        # 8. Build grid
        _progress(7, PipelineStage.BUILDING_GRID)
        n_il_orig = len(inlines_3d)
        n_xl_orig = len(xlines_3d)
        p00 = coords_3d[0]
        p0n = coords_3d[n_xl_orig - 1]
        pn0 = coords_3d[(n_il_orig - 1) * n_xl_orig]
        il_step_xy = (pn0 - p00) / max(n_il_orig - 1, 1)
        xl_step_xy = (p0n - p00) / max(n_xl_orig - 1, 1)

        vb = VolumeBuilder(
            geometry=geom, orig_inlines=inlines_3d, orig_xlines=xlines_3d,
            n_samples=meta3d.n_samples, dt_ms=meta3d.dt_ms, expand_polygon=expand_poly,
            origin_x=float(p00[0]), origin_y=float(p00[1]),
            il_step_x=float(il_step_xy[0]), il_step_y=float(il_step_xy[1]),
            xl_step_x=float(xl_step_xy[0]), xl_step_y=float(xl_step_xy[1]),
        )
        target_grid = vb.build_target_grid(max_volume_gb=cfg.max_grid_memory_gb)
        st.session_state.target_grid = target_grid
        sparse = vb.inject_lines(target_grid, calibrated_lines)
        orig_in_grid, orig_mask = vb.inject_original_3d(target_grid, cube_volume, inlines_3d, xlines_3d)
        del cube_volume; import gc; gc.collect()  # free ~0.15-5.5 GB
        st.session_state.orig_mask = orig_mask
        st.session_state.sparse_mask = sparse.mask.copy()
        st.session_state.sparse_fill = float(sparse.fill_ratio)

        # 9. Tune interpolation
        _progress(8, PipelineStage.TUNING_INTERP)
        interpolator = _make_interpolator(cfg)
        sim_mask = runner._create_simulation_mask(orig_mask, sparse.mask, inlines_3d, xlines_3d, target_grid)
        sim_metrics = interpolator.fit(orig_in_grid, sim_mask)
        st.session_state.interp_sim_metrics = sim_metrics

        # 10. Reconstruct
        _progress(9, PipelineStage.RECONSTRUCTING)
        combined_data = orig_in_grid.copy()
        combined_mask = orig_mask.copy()
        ext_only = sparse.mask & ~orig_mask
        combined_data[ext_only] = np.nan_to_num(sparse.data[ext_only], nan=0.0)
        combined_mask[ext_only] = True
        del sparse; gc.collect()

        from seis2cube.models.volume import SparseVolume as SV
        combined_sparse = SV(grid=target_grid, data=combined_data, mask=combined_mask)
        result = interpolator.reconstruct(combined_sparse)
        del combined_sparse; gc.collect()
        st.session_state.recon_volume = result.volume

        # 11. Assemble
        _progress(10, PipelineStage.ASSEMBLING)
        final = VolumeBuilder.assemble(
            orig_vol=orig_in_grid, orig_mask=orig_mask, recon_vol=result.volume,
            taper_width=cfg.blend.taper_width_traces if cfg.blend.enabled else 0,
            blend=cfg.blend.enabled,
        )
        st.session_state.final_volume = final
        del orig_in_grid, result; gc.collect()

        # 12. Write
        _progress(11, PipelineStage.WRITING)
        writer = SegyWriter3D(
            path=cfg.out_cube_path, inlines=target_grid.inlines, xlines=target_grid.xlines,
            dt_us=int(meta3d.dt_ms * 1000), header_bytes=cfg.header_bytes,
            origin_x=target_grid.origin_x, origin_y=target_grid.origin_y,
            il_step_x=target_grid.il_step_x, il_step_y=target_grid.il_step_y,
            xl_step_x=target_grid.xl_step_x, xl_step_y=target_grid.xl_step_y,
            delrt_ms=meta3d.delrt_ms,
        )
        out_path = writer.write(final)

        # 13. QC
        _progress(12, PipelineStage.QC)
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

        # Done
        set_stage(PipelineStage.DONE, 1.0)
        progress_bar.progress(1.0)
        status_text.markdown(f"**✅ Complete! → `{out_path}`**")
        add_log(f"Output: {out_path}")

    except Exception as e:
        set_stage(PipelineStage.ERROR)
        st.session_state.error_msg = str(e)
        add_log(f"ERROR: {e}")
        status_text.markdown(f"**❌ {e}**")
        import traceback
        traceback.print_exc()


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
        with st.expander("⚙️ Active Configuration", expanded=False):
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
            st.plotly_chart(fig, use_container_width=True)


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
        # Split info
        n_train = getattr(st.session_state, "n_train", "?")
        n_test = getattr(st.session_state, "n_test", "?")
        st.caption(
            f"📊 Train: **{n_train}** pairs · Test (holdout): **{n_test}** pairs · "
            f"All metrics and plots below are on the **test set** (data NOT used for fitting)"
        )

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
            st.plotly_chart(fig_scatter, use_container_width=True)

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
            st.plotly_chart(fig_strip, use_container_width=True)

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

            fig = plot_trace_comparison(
                traces_2d[trace_idx], traces_3d[trace_idx], traces_corr[trace_idx],
                time_axis=time_ax,
                title=f"Test Pair #{trace_idx + 1} — 2D vs 3D vs Corrected",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Spectrum comparison
            from seis2cube.utils.spectral import amplitude_spectrum
            freqs, sp2d = amplitude_spectrum(traces_2d[trace_idx], dt)
            _, sp3d = amplitude_spectrum(traces_3d[trace_idx], dt)
            _, spcorr = amplitude_spectrum(traces_corr[trace_idx], dt)

            fig_sp = plot_spectrum_comparison(freqs, sp2d, sp3d, spcorr,
                                             title=f"Spectrum — Test Pair #{trace_idx + 1}")
            st.plotly_chart(fig_sp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB: Interpolation
# ════════════════════════════════════════════════════════════════════════

with tab_interp:
    section_header("Interpolation Results", "🧩")

    sim = st.session_state.interp_sim_metrics

    if sim is None:
        st.info("Run the pipeline to see interpolation results.")
    else:
        import plotly.graph_objects as go

        # ── Explanation ──────────────────────────────────────────────
        st.markdown("""
**Как оценивается качество интерполяции?**

Внутри зоны оригинального 3D куба мы знаем истинные значения. Pipeline «прячет» часть трасс,
восстанавливает их интерполятором, и сравнивает с оригиналом. Это **симуляция** — она
показывает, насколько хорошо интерполятор заполняет пробелы, когда у него есть только
разреженные 2D профили.
        """)

        # ── Grid overview ────────────────────────────────────────────
        grid = st.session_state.target_grid
        orig_mask = st.session_state.orig_mask
        sparse_mask = getattr(st.session_state, "sparse_mask", None)

        if grid is not None:
            st.markdown("### Расширенный грид")
            n_total = grid.n_il * grid.n_xl
            n_orig = int(orig_mask.sum()) if orig_mask is not None else 0
            n_2d = int(sparse_mask.sum()) if sparse_mask is not None else 0
            n_ext_2d = int((sparse_mask & ~orig_mask).sum()) if sparse_mask is not None and orig_mask is not None else 0
            n_empty = n_total - n_orig - n_ext_2d
            pct_orig = 100 * n_orig / max(n_total, 1)
            pct_2d = 100 * n_ext_2d / max(n_total, 1)
            pct_empty = 100 * n_empty / max(n_total, 1)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Grid Size", f"{grid.n_il} × {grid.n_xl}", icon="�️")
            with c2:
                render_metric_card("3D Original", f"{n_orig:,} ({pct_orig:.1f}%)", icon="�")
            with c3:
                render_metric_card("2D Profiles", f"{n_ext_2d:,} ({pct_2d:.1f}%)", icon="📏")
            with c4:
                render_metric_card("To Interpolate", f"{n_empty:,} ({pct_empty:.1f}%)", icon="🧩")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Simulation quality metrics ───────────────────────────────
        st.markdown("### Качество реконструкции (симуляция)")
        st.caption(
            "Метрики рассчитаны на тех позициях внутри 3D куба, где данные были «спрятаны» "
            "и восстановлены интерполятором. Чем ниже RMSE/MAE и выше корреляция — тем лучше."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card(
                "RMSE",
                f"{sim['rmse']:.1f}",
                icon="📐",
                delta="среднеквадр. ошибка",
            )
        with c2:
            render_metric_card(
                "MAE",
                f"{sim['mae']:.1f}",
                icon="📏",
                delta="средняя абс. ошибка",
            )
        with c3:
            corr_val = sim["pearson_corr"]
            quality = "отлично" if corr_val > 0.8 else "хорошо" if corr_val > 0.5 else "слабо"
            render_metric_card(
                "Корреляция",
                f"{corr_val:.4f}",
                icon="🎯",
                delta=quality,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Coverage map: 3-colour heatmap ───────────────────────────
        if orig_mask is not None:
            st.markdown("### Карта покрытия")
            st.caption(
                "🟦 Синий = оригинальный 3D куб · 🟩 Зелёный = 2D профили · "
                "⬛ Тёмный = пусто (заполнено интерполяцией)"
            )

            # Build 3-value map: 0=empty, 1=2D-only, 2=original-3D
            coverage = np.zeros_like(orig_mask, dtype=np.float32)
            if sparse_mask is not None:
                ext_2d = sparse_mask & ~orig_mask
                coverage[ext_2d] = 1.0
            coverage[orig_mask] = 2.0

            fig_cov = go.Figure(data=go.Heatmap(
                z=coverage,
                colorscale=[
                    [0.0, PALETTE["bg_dark"]],
                    [0.5, "#a6e3a1"],
                    [1.0, PALETTE["primary"]],
                ],
                zmin=0, zmax=2,
                showscale=False,
                hovertemplate="IL idx: %{y}<br>XL idx: %{x}<br>Type: %{z}<extra></extra>",
            ))
            fig_cov.update_layout(
                height=450,
                paper_bgcolor=PALETTE["bg_dark"],
                plot_bgcolor=PALETTE["bg_dark"],
                font=dict(color=PALETTE["text"]),
                xaxis_title="Crossline index",
                yaxis_title="Inline index",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=60, r=20, t=30, b=50),
            )
            st.plotly_chart(fig_cov, use_container_width=True)

            # ── Fill ratio pie chart ─────────────────────────────────
            st.markdown("### Соотношение данных")
            labels = ["3D Original", "2D Profiles", "Interpolated"]
            values = [n_orig, n_ext_2d, n_empty]
            colors_pie = [PALETTE["primary"], "#a6e3a1", "#45475a"]

            fig_pie = go.Figure(data=go.Pie(
                labels=labels, values=values,
                marker=dict(colors=colors_pie),
                textinfo="label+percent",
                textfont=dict(size=13, color="white"),
                hole=0.4,
            ))
            fig_pie.update_layout(
                height=350,
                paper_bgcolor=PALETTE["bg_card"],
                font=dict(color=PALETTE["text"]),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB: Volume Viewer
# ════════════════════════════════════════════════════════════════════════

with tab_viewer:
    section_header("Volume Viewer", "🗺️")

    vol = st.session_state.final_volume
    orig_vol = st.session_state.cube_volume

    if vol is None:
        st.info("Run the pipeline to view volumes.")
    else:
        grid = st.session_state.target_grid

        view_mode = st.radio(
            "View",
            ["Time Slice", "Inline Section", "Crossline Section"],
            horizontal=True,
            key="viewer_mode",
        )

        volume_sel = st.radio(
            "Volume",
            ["Final (extended)", "Original 3D", "Side by Side"],
            horizontal=True,
            key="viewer_vol",
        )

        if view_mode == "Time Slice":
            max_samp = vol.shape[2] - 1
            t_idx = st.slider("Sample index", 0, max_samp, max_samp // 3, key="ts_slider")
            _delrt = st.session_state.meta_3d.get("delrt_ms", 0.0) if st.session_state.meta_3d else 0.0
            t_ms = _delrt + t_idx * (grid.dt_ms if grid else 2.0)
            st.caption(f"Time: {t_ms:.1f} ms")

            if volume_sel == "Side by Side" and orig_vol is not None:
                c1, c2 = st.columns(2)
                with c1:
                    fig_o = plot_time_slice(orig_vol, t_idx, title="Original 3D", height=400)
                    st.plotly_chart(fig_o, use_container_width=True)
                with c2:
                    fig_f = plot_time_slice(vol, t_idx,
                                           inlines=grid.inlines if grid else None,
                                           xlines=grid.xlines if grid else None,
                                           title="Extended", height=400)
                    st.plotly_chart(fig_f, use_container_width=True)
            else:
                show_vol = vol if "Final" in volume_sel else (orig_vol if orig_vol is not None else vol)
                label = "Extended" if "Final" in volume_sel else "Original"
                fig = plot_time_slice(show_vol, t_idx, title=f"{label} — t={t_ms:.1f} ms")
                st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "Inline Section":
            max_il = vol.shape[0] - 1
            il_idx = st.slider("Inline index", 0, max_il, max_il // 2, key="il_slider")
            _delrt = st.session_state.meta_3d.get("delrt_ms", 0.0) if st.session_state.meta_3d else 0.0
            time_ax = _delrt + np.arange(vol.shape[2]) * (grid.dt_ms if grid else 2.0)

            if volume_sel == "Side by Side" and orig_vol is not None:
                c1, c2 = st.columns(2)
                with c1:
                    il_o = min(il_idx, orig_vol.shape[0] - 1)
                    fig = plot_inline_section(orig_vol, il_o, time_axis=time_ax[:orig_vol.shape[2]],
                                             title="Original", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = plot_inline_section(vol, il_idx, time_axis=time_ax, title="Extended", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                show_vol = vol if "Final" in volume_sel else (orig_vol if orig_vol is not None else vol)
                idx = min(il_idx, show_vol.shape[0] - 1)
                fig = plot_inline_section(show_vol, idx, time_axis=time_ax[:show_vol.shape[2]])
                st.plotly_chart(fig, use_container_width=True)

        else:  # Crossline
            max_xl = vol.shape[1] - 1
            xl_idx = st.slider("Crossline index", 0, max_xl, max_xl // 2, key="xl_slider")
            _delrt = st.session_state.meta_3d.get("delrt_ms", 0.0) if st.session_state.meta_3d else 0.0
            time_ax = _delrt + np.arange(vol.shape[2]) * (grid.dt_ms if grid else 2.0)

            show_vol = vol if "Final" in volume_sel else (orig_vol if orig_vol is not None else vol)
            idx = min(xl_idx, show_vol.shape[1] - 1)
            section = show_vol[:, idx, :]
            vmax = np.percentile(np.abs(section[np.isfinite(section)]), 98) if np.any(np.isfinite(section)) else 1.0
            vmax = max(vmax, 1e-10)

            import plotly.graph_objects as go
            fig = go.Figure(data=go.Heatmap(
                z=section.T, y=time_ax[:section.shape[1]],
                colorscale=SEISMIC_COLORSCALE,
                zmin=-vmax, zmax=vmax,
            ))
            fig.update_layout(
                title=f"Crossline #{xl_idx}",
                xaxis_title="Inline index", yaxis_title="Time (ms)",
                height=500, paper_bgcolor=PALETTE["bg_dark"],
                plot_bgcolor=PALETTE["bg_dark"], font=dict(color=PALETTE["text"]),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)


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

        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
