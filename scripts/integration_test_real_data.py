#!/usr/bin/env python3
"""Integration test on real data — exercises pipeline steps + UI components.

Tests the same code paths that app.py exercises, generating Plotly figures
via seis2cube.ui.components, then saves them to an interactive HTML report.

**Memory-safe**: keeps the 3D SEG-Y open with mmap, reads inlines on demand,
NEVER loads the full 5.5 GB volume into RAM.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import plotly.io as pio

from seis2cube.config import IOConfig, PipelineConfig, SegyHeaderBytes
from seis2cube.io.segy_dataset import SegyDataset
from seis2cube.geometry.crs_converter import CRSConverter
from seis2cube.geometry.geometry_model import AffineGridMapper
from seis2cube.geometry.overlap_detector import OverlapDetector
from seis2cube.models.line2d import Line2D
from seis2cube.calibration.global_shift import GlobalShiftGainPhase
from seis2cube.calibration.base import CalibrationPair
from seis2cube.pipeline.runner import PipelineRunner
from seis2cube.ui.components import (
    PALETTE,
    SEISMIC_COLORSCALE,
    plot_inline_section,
    plot_map_with_lines,
    plot_metrics_radar,
    plot_spectrum_comparison,
    plot_time_slice,
    plot_trace_comparison,
)
from seis2cube.utils.spectral import amplitude_spectrum

CFG_PATH = Path("configs/test_data2.yaml")
OUT_HTML = Path("scripts/integration_report.html")

timings: dict[str, float] = {}
figs: list[tuple[str, go.Figure]] = []


def timed(name: str):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"▶ {name} ...", flush=True)
            return self
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self.t0
            timings[name] = elapsed
            print(f"  ✓ {name}: {elapsed:.2f}s", flush=True)
    return _T()


def mem_mb():
    """Current RSS in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading config")
print("=" * 70)
cfg = PipelineConfig.from_yaml(CFG_PATH)
runner = PipelineRunner(cfg)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Ingest 3D (metadata + mmap, NO full volume load)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Ingest 3D cube (mmap, no full load)")
print("=" * 70)

with timed("Open 3D + metadata"):
    ds3d = SegyDataset(cfg.cube3d_path, cfg.header_bytes, cfg.io)
    ds3d.open()
    meta3d = ds3d.meta
    coords_3d = ds3d.all_coordinates()
    n_il = len(meta3d.inlines)
    n_xl = len(meta3d.xlines)
    print(f"    {n_il} IL × {n_xl} XL × {meta3d.n_samples} samp")
    print(f"    dt={meta3d.dt_ms}ms  delrt={meta3d.delrt_ms}ms")
    print(f"    RSS: {mem_mb():.0f} MB")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: CRS + Geometry + Overlap
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: CRS + Geometry + Overlap")
print("=" * 70)

with timed("CRS conversion"):
    crs_conv = CRSConverter(cfg.crs)
    if not crs_conv.is_identity:
        cx, cy = crs_conv.forward(coords_3d[:, 0], coords_3d[:, 1])
        coords_3d = np.column_stack([cx, cy])

with timed("AffineGridMapper"):
    geom = AffineGridMapper(coords_3d, meta3d.inlines, meta3d.xlines)

with timed("OverlapDetector"):
    expand_poly = OverlapDetector.load_polygon(cfg.expand_polygon_path)
    overlap = OverlapDetector.from_3d_coords(coords_3d, expand_polygon=expand_poly)

print(f"    RSS: {mem_mb():.0f} MB")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Load all 2D lines
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Load 2D lines")
print("=" * 70)

lines_2d: list[Line2D] = []
with timed("Load + resample 2D lines"):
    for lpath in cfg.lines2d_paths:
        line = runner._load_2d_line(
            lpath, crs_conv, meta3d.dt_ms, meta3d.n_samples,
            target_delrt_ms=meta3d.delrt_ms,
        )
        lines_2d.append(line)
        print(f"    {line.name}: {line.n_traces} traces, "
              f"delrt={line.delrt_ms:.0f}ms, dt={line.dt_ms}ms")

total_2d = sum(l.n_traces for l in lines_2d)
print(f"    Total: {len(lines_2d)} lines, {total_2d:,} traces")
print(f"    RSS: {mem_mb():.0f} MB")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Overlap classification
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Overlap classification")
print("=" * 70)

line_inside_counts = {}
with timed("Classify 2D traces"):
    for line in lines_2d:
        inside, expand, outside = overlap.classify_line(line.coords)
        n_in = int(inside.sum())
        line_inside_counts[line.name] = n_in
        print(f"    {line.name}: inside={n_in}, expand={int(expand.sum())}, "
              f"outside={int(outside.sum())}")

# ═══════════════════════════════════════════════════════════════════════
# FIG 1: Survey Map (using UI component plot_map_with_lines)
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Fig 1: Survey Map via plot_map_with_lines ...", flush=True)

poly_coords = np.array(expand_poly.exterior.coords)
fig_map = plot_map_with_lines(
    coords_3d,
    [l.coords for l in lines_2d],
    [l.name for l in lines_2d],
    polygon_xy=poly_coords,
    title="Survey Map: 3D Coverage + 2D Lines + Expansion Polygon",
)
hull_poly = overlap.cube_polygon
hx, hy = hull_poly.exterior.xy
fig_map.add_trace(go.Scatter(
    x=list(hx), y=list(hy), mode="lines",
    line=dict(color=PALETTE["success"], width=2, dash="dash"),
    name="3D hull",
))
figs.append(("Survey Map", fig_map))
print("  ✓ done")

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Build calibration pairs — ON DEMAND from mmap (no full cube)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Calibration pairs (inline-on-demand, no full cube load)")
print("=" * 70)


def _build_pairs_on_demand(
    lines, overlap, geom, ds3d, meta3d,
):
    """Build calibration pairs reading inlines from mmap on demand."""
    inlines_3d = meta3d.inlines
    xlines_3d = meta3d.xlines
    n_il = len(inlines_3d)
    n_xl = len(xlines_3d)
    n_samp = meta3d.n_samples

    # Cache: inline_number -> (n_xl, n_samp) array
    _il_cache: dict[int, np.ndarray] = {}

    def _get_trace(il_val: int, xl_idx: int) -> np.ndarray:
        if il_val not in _il_cache:
            # Evict oldest to keep cache small (max ~20 inlines = ~40 MB)
            if len(_il_cache) > 20:
                _il_cache.pop(next(iter(_il_cache)))
            _il_cache[il_val] = ds3d.read_inline(il_val)
        return _il_cache[il_val][xl_idx]

    all_coords, all_2d, all_3d = [], [], []

    for line in lines:
        inside_idx = overlap.overlap_indices(line.coords)
        if len(inside_idx) == 0:
            continue
        for idx in inside_idx:
            x, y = line.coords[idx]
            il_frac, xl_frac = geom.xy_to_ilxl(np.array([x]), np.array([y]))
            il_near = int(round(il_frac[0]))
            xl_near = int(round(xl_frac[0]))
            il_idx_arr = np.searchsorted(inlines_3d, il_near)
            xl_idx_arr = np.searchsorted(xlines_3d, xl_near)
            if il_idx_arr >= n_il or xl_idx_arr >= n_xl:
                continue
            if inlines_3d[il_idx_arr] != il_near or xlines_3d[xl_idx_arr] != xl_near:
                continue  # skip non-exact matches for speed
            trace_3d = _get_trace(il_near, xl_idx_arr)
            all_coords.append([x, y])
            all_2d.append(line.data[idx])
            all_3d.append(trace_3d)

    if not all_2d:
        raise RuntimeError("No calibration pairs found!")

    coords_arr = np.array(all_coords)
    amp_2d = np.array(all_2d, dtype=np.float32)
    amp_3d = np.array(all_3d, dtype=np.float32)
    n = len(amp_2d)
    holdout = cfg.calibration.holdout_fraction
    n_test = max(1, int(n * holdout))
    train = CalibrationPair(coords=coords_arr[:-n_test], amp_2d=amp_2d[:-n_test],
                            amp_3d=amp_3d[:-n_test], dt_ms=meta3d.dt_ms)
    test = CalibrationPair(coords=coords_arr[-n_test:], amp_2d=amp_2d[-n_test:],
                           amp_3d=amp_3d[-n_test:], dt_ms=meta3d.dt_ms)
    return train, test


with timed("Build calibration pairs (on-demand)"):
    train_pairs, test_pairs = _build_pairs_on_demand(
        lines_2d, overlap, geom, ds3d, meta3d,
    )
    print(f"    Train: {train_pairs.amp_2d.shape[0]}, Test: {test_pairs.amp_2d.shape[0]}")
    print(f"    RSS: {mem_mb():.0f} MB")

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Calibrate
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: GlobalShiftGainPhase calibration")
print("=" * 70)

with timed("Calibration fit"):
    calibrator = GlobalShiftGainPhase(
        max_shift_ms=50.0, estimate_phase=True,
        estimate_matching_filter=False,
    )
    cal_model = calibrator.fit(train_pairs)
    print(f"    Model params: {cal_model.params}")

with timed("Calibration evaluate"):
    metrics_after = calibrator.evaluate(test_pairs, cal_model)
    print(f"    Test metrics: {metrics_after}")

with timed("Apply calibration to test"):
    corrected_test = calibrator._apply_array(test_pairs.amp_2d, cal_model)

# Before/after correlation
corrs_before = []
for i in range(len(test_pairs.amp_2d)):
    c = np.corrcoef(test_pairs.amp_2d[i], test_pairs.amp_3d[i])[0, 1]
    if np.isfinite(c):
        corrs_before.append(c)
corrs_before = np.array(corrs_before)

corrs_after = []
for i in range(len(corrected_test)):
    c = np.corrcoef(corrected_test[i], test_pairs.amp_3d[i])[0, 1]
    if np.isfinite(c):
        corrs_after.append(c)
corrs_after = np.array(corrs_after)

print(f"    Correlation: {corrs_before.mean():.4f} → {corrs_after.mean():.4f}")
baseline_rmse = float(np.sqrt(np.mean((test_pairs.amp_2d - test_pairs.amp_3d) ** 2)))
print(f"    RMSE:        {baseline_rmse:.2f} → {metrics_after['rmse']:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# FIG 2-5: Trace comparisons via plot_trace_comparison (UI component)
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Fig 2-5: Trace comparisons via plot_trace_comparison ...", flush=True)

n_show = min(4, len(test_pairs.amp_2d))
t_axis = meta3d.delrt_ms + np.arange(meta3d.n_samples) * meta3d.dt_ms

for i in range(n_show):
    fig_tc = plot_trace_comparison(
        test_pairs.amp_2d[i], test_pairs.amp_3d[i], corrected_test[i],
        time_axis=t_axis,
        title=f"Trace #{i+1} — 2D vs 3D vs Calibrated "
              f"(r={corrs_before[i]:.3f}→{corrs_after[i]:.3f})",
    )
    figs.append((f"Trace Comparison #{i+1}", fig_tc))
print("  ✓ done")

# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Spectrum comparison via plot_spectrum_comparison (UI component)
# ═══════════════════════════════════════════════════════════════════════
print("▶ Fig 6: Spectrum comparison via plot_spectrum_comparison ...", flush=True)

n_spec = min(200, len(test_pairs.amp_2d))
freqs, sp_2d = amplitude_spectrum(test_pairs.amp_2d[:n_spec], meta3d.dt_ms)
_, sp_3d = amplitude_spectrum(test_pairs.amp_3d[:n_spec], meta3d.dt_ms)
_, sp_cal = amplitude_spectrum(corrected_test[:n_spec], meta3d.dt_ms)

fig_spec = plot_spectrum_comparison(
    freqs, sp_2d.mean(axis=0), sp_3d.mean(axis=0), sp_cal.mean(axis=0),
    title="Mean Amplitude Spectrum: 2D vs 3D vs Calibrated",
)
figs.append(("Spectrum Comparison", fig_spec))
print("  ✓ done")

# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Metrics radar via plot_metrics_radar (UI component)
# ═══════════════════════════════════════════════════════════════════════
print("▶ Fig 7: Metrics radar via plot_metrics_radar ...", flush=True)

fig_radar = plot_metrics_radar(
    {
        "Correlation": max(metrics_after["pearson_corr"], 0),
        "1-RMSE": max(1 - metrics_after["rmse"] / max(baseline_rmse, 1e-9), 0),
        "1-MAE": max(1 - metrics_after["mae"] / max(baseline_rmse, 1e-9), 0),
        "1-SpecDiff": max(1 - metrics_after["spectral_l2_rel"], 0),
    },
    title="Calibration Quality (higher = better)",
)
figs.append(("Quality Radar", fig_radar))
print("  ✓ done")

# ═══════════════════════════════════════════════════════════════════════
# FIG 8: Inline section — read single inline from mmap (< 2 MB)
# ═══════════════════════════════════════════════════════════════════════
print("▶ Fig 8: Inline section (single inline from mmap) ...", flush=True)

mid_il_val = int(meta3d.inlines[n_il // 2])
il_data = ds3d.read_inline(mid_il_val)  # (n_xl, n_samp) — ~1.7 MB
time_ax_sec = meta3d.delrt_ms + np.arange(meta3d.n_samples) * meta3d.dt_ms

# plot_inline_section expects (n_il, n_xl, n_samp) volume, fake with 1 inline
vol_1il = il_data[np.newaxis, :, :]  # (1, n_xl, n_samp)
fig_isec = plot_inline_section(
    vol_1il, 0,
    xlines=meta3d.xlines.astype(float),
    time_axis=time_ax_sec,
    title=f"3D Inline {mid_il_val} Section",
)
figs.append(("Inline Section", fig_isec))
del il_data, vol_1il
print("  ✓ done")

# ═══════════════════════════════════════════════════════════════════════
# FIG 9: Time slice — read single depth slice from mmap (< 38 MB)
# ═══════════════════════════════════════════════════════════════════════
print("▶ Fig 9: Time slice (single depth_slice from mmap) ...", flush=True)

mid_t = meta3d.n_samples // 3
t_ms = meta3d.delrt_ms + mid_t * meta3d.dt_ms
ts_data = ds3d.read_time_slice(mid_t)  # (n_il, n_xl) — ~38 MB

# plot_time_slice expects (n_il, n_xl, n_samp) volume; fake with 1 sample
vol_1t = ts_data[:, :, np.newaxis]  # (n_il, n_xl, 1)
fig_ts = plot_time_slice(
    vol_1t, 0,
    inlines=meta3d.inlines.astype(float),
    xlines=meta3d.xlines.astype(float),
    title=f"3D Time Slice at {t_ms:.0f} ms",
)
figs.append(("Time Slice", fig_ts))
del ts_data, vol_1t
print("  ✓ done")

# ═══════════════════════════════════════════════════════════════════════
# FIG 10: Amplitude cross-plot + regression
# ═══════════════════════════════════════════════════════════════════════
print("▶ Fig 10: Amplitude cross-plot + regression ...", flush=True)

flat_2d = test_pairs.amp_2d.ravel()
flat_3d = test_pairs.amp_3d.ravel()
max_pts = 50_000
rng = np.random.default_rng(42)
if len(flat_2d) > max_pts:
    sub = rng.choice(len(flat_2d), max_pts, replace=False)
    s2d, s3d = flat_2d[sub], flat_3d[sub]
else:
    s2d, s3d = flat_2d, flat_3d

valid = np.isfinite(s2d) & np.isfinite(s3d) & (s2d != 0) & (s3d != 0)
coeffs = np.polyfit(s2d[valid], s3d[valid], 1) if valid.sum() > 10 else [1, 0]
corr_all = np.corrcoef(s2d[valid], s3d[valid])[0, 1] if valid.sum() > 10 else 0
fit_x = np.array([s2d[valid].min(), s2d[valid].max()])
fit_y = np.polyval(coeffs, fit_x)

fig_xp = go.Figure()
fig_xp.add_trace(go.Scattergl(
    x=s2d[valid], y=s3d[valid], mode="markers",
    marker=dict(size=1, color="rgba(99,102,241,0.15)"), name="samples",
))
fig_xp.add_trace(go.Scatter(
    x=fit_x, y=fit_y, mode="lines",
    line=dict(color=PALETTE["danger"], width=2),
    name=f"fit: y={coeffs[0]:.3f}x + {coeffs[1]:.1f}",
))
fig_xp.add_trace(go.Scatter(
    x=fit_x, y=fit_x, mode="lines",
    line=dict(color=PALETTE["success"], width=1, dash="dash"), name="y=x",
))
fig_xp.update_layout(
    title=f"Amplitude Cross-plot (r={corr_all:.4f}, gain={coeffs[0]:.3f})",
    xaxis_title="2D amplitude", yaxis_title="3D amplitude",
    height=550, width=650,
    paper_bgcolor=PALETTE["bg_dark"], plot_bgcolor=PALETTE["bg_dark"],
    font=dict(color=PALETTE["text"]),
)
figs.append(("Amplitude Cross-plot", fig_xp))
print(f"  ✓ done (r={corr_all:.4f})")

# ═══════════════════════════════════════════════════════════════════════
# FIG 11: Per-trace correlation histogram (before/after)
# ═══════════════════════════════════════════════════════════════════════
print("▶ Fig 11: Correlation histogram ...", flush=True)

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=corrs_before, nbinsx=50, opacity=0.6,
    marker_color=PALETTE["danger"], name="Before calibration",
))
fig_hist.add_trace(go.Histogram(
    x=corrs_after, nbinsx=50, opacity=0.6,
    marker_color=PALETTE["success"], name="After calibration",
))
fig_hist.update_layout(
    barmode="overlay",
    title=f"Per-trace Correlation (test set): "
          f"{corrs_before.mean():.3f} → {corrs_after.mean():.3f}",
    xaxis_title="Pearson r", yaxis_title="Count",
    height=400, width=650,
    paper_bgcolor=PALETTE["bg_dark"], plot_bgcolor=PALETTE["bg_dark"],
    font=dict(color=PALETTE["text"]),
)
figs.append(("Correlation Histogram", fig_hist))
print("  ✓ done")

# Close 3D SEG-Y — done with mmap reads
ds3d.close()
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# FIG 12: Timing breakdown
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Fig 12: Timing breakdown ...", flush=True)

names_t = list(timings.keys())
times_t = list(timings.values())
fig_timing = go.Figure(data=go.Bar(
    x=times_t, y=names_t, orientation="h",
    marker_color=PALETTE["primary"],
    text=[f"{t:.2f}s" for t in times_t], textposition="outside",
))
fig_timing.update_layout(
    title=f"Timing Breakdown (total: {sum(times_t):.1f}s)",
    xaxis_title="Seconds", height=max(300, 40 * len(names_t)), width=700,
    yaxis=dict(autorange="reversed"),
    paper_bgcolor=PALETTE["bg_dark"], plot_bgcolor=PALETTE["bg_dark"],
    font=dict(color=PALETTE["text"]),
)
figs.append(("Timing", fig_timing))

# ═══════════════════════════════════════════════════════════════════════
# SAVE HTML
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Saving report ...")
print("=" * 70)

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_HTML, "w") as fh:
    fh.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>seis2cube Integration Report</title>
<style>
body {{ font-family: 'Inter', sans-serif; margin: 20px; background: {PALETTE['bg_dark']}; color: {PALETTE['text']}; }}
h1 {{ color: {PALETTE['text']}; }}
h2 {{ color: {PALETTE['text_muted']}; margin-top: 30px; }}
.summary {{ background: {PALETTE['bg_card']}; padding: 15px 20px; border-radius: 12px;
           border: 1px solid #313244; margin-bottom: 20px; }}
hr {{ margin: 30px 0; border: none; border-top: 1px solid #313244; }}
</style>
</head><body>
<h1>seis2cube Integration Test Report</h1>
<div class="summary">
<b>3D Cube:</b> {n_il} IL × {n_xl} XL × {meta3d.n_samples} samp,
dt={meta3d.dt_ms}ms, delrt={meta3d.delrt_ms}ms<br>
<b>2D Lines:</b> {len(lines_2d)} lines, {total_2d:,} traces<br>
<b>Calibration pairs:</b> train={train_pairs.amp_2d.shape[0]}, test={test_pairs.amp_2d.shape[0]}<br>
<b>Cal. model:</b> {cal_model.params}<br>
<b>Correlation:</b> {corrs_before.mean():.4f} → {corrs_after.mean():.4f}<br>
<b>RMSE:</b> {baseline_rmse:.2f} → {metrics_after['rmse']:.2f}<br>
<b>Peak RSS:</b> {mem_mb():.0f} MB<br>
<b>Total time:</b> {sum(times_t):.1f}s
</div>
""")
    for i, (title, fig) in enumerate(figs):
        fh.write(f"<h2>{title}</h2>\n")
        fh.write(pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0)))
        fh.write("\n<hr>\n")
    fh.write("</body></html>")

print(f"\n✅ Report saved: {OUT_HTML}")
print(f"   Peak RSS: {mem_mb():.0f} MB")
print(f"   Total time: {sum(times_t):.1f}s")
print(f"   Correlation: {corrs_before.mean():.4f} → {corrs_after.mean():.4f}")
print(f"   RMSE:        {baseline_rmse:.2f} → {metrics_after['rmse']:.2f}")
