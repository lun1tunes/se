"""Reusable UI components for the Streamlit app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


# ── Colour palette ──────────────────────────────────────────────────────

PALETTE = {
    "primary": "#6366F1",      # indigo-500
    "secondary": "#8B5CF6",    # violet-500
    "success": "#10B981",      # emerald-500
    "warning": "#F59E0B",      # amber-500
    "danger": "#EF4444",       # red-500
    "bg_card": "#1E1E2E",
    "bg_dark": "#11111B",
    "text": "#CDD6F4",
    "text_muted": "#6C7086",
    "seismic_low": "#2563EB",
    "seismic_mid": "#FFFFFF",
    "seismic_high": "#DC2626",
}

SEISMIC_COLORSCALE = [
    [0.0, "#2563EB"],
    [0.25, "#60A5FA"],
    [0.5, "#FFFFFF"],
    [0.75, "#F87171"],
    [1.0, "#DC2626"],
]


# ── Status / Progress ──────────────────────────────────────────────────

def render_status_badge(label: str, status: str = "info") -> None:
    """Render a coloured status badge."""
    colours = {
        "info": PALETTE["primary"],
        "success": PALETTE["success"],
        "warning": PALETTE["warning"],
        "error": PALETTE["danger"],
        "running": PALETTE["secondary"],
    }
    colour = colours.get(status, PALETTE["primary"])
    st.markdown(
        f'<span style="background:{colour}; color:white; padding:4px 12px; '
        f'border-radius:12px; font-size:0.85rem; font-weight:600;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, delta: str | None = None, icon: str = "") -> None:
    """Render a single metric in a styled card."""
    delta_html = ""
    if delta is not None:
        is_positive = delta.startswith("+")
        colour = PALETTE["success"] if is_positive else PALETTE["danger"]
        delta_html = f'<span style="color:{colour}; font-size:0.85rem;">{delta}</span>'

    st.markdown(
        f"""
        <div style="background:{PALETTE['bg_card']}; border-radius:12px; padding:20px;
                    border:1px solid #313244;">
            <div style="color:{PALETTE['text_muted']}; font-size:0.8rem; text-transform:uppercase;
                        letter-spacing:0.05em; margin-bottom:6px;">
                {icon} {title}
            </div>
            <div style="color:{PALETTE['text']}; font-size:1.8rem; font-weight:700;">
                {value}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_progress(stage_label: str, progress: float, stages_done: int, total: int) -> None:
    """Render a multi-step progress indicator."""
    st.markdown(
        f"""
        <div style="background:{PALETTE['bg_card']}; border-radius:12px; padding:20px;
                    border:1px solid #313244; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:{PALETTE['text']}; font-weight:600;">{stage_label}</span>
                <span style="color:{PALETTE['text_muted']};">Step {stages_done}/{total}</span>
            </div>
            <div style="background:#313244; border-radius:8px; height:10px; overflow:hidden;">
                <div style="background:linear-gradient(90deg, {PALETTE['primary']}, {PALETTE['secondary']});
                            width:{progress*100:.0f}%; height:100%; border-radius:8px;
                            transition:width 0.4s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Seismic visualisation ──────────────────────────────────────────────

def plot_time_slice(
    data: np.ndarray,
    sample_idx: int,
    inlines: np.ndarray | None = None,
    xlines: np.ndarray | None = None,
    title: str = "Time Slice",
    height: int = 500,
) -> go.Figure:
    """Plot a single time-slice as a heatmap."""
    if data.ndim == 3:
        slice_data = data[:, :, sample_idx]
    else:
        slice_data = data

    finite_vals = slice_data[np.isfinite(slice_data)]
    vmax = float(np.percentile(np.abs(finite_vals), 98)) if len(finite_vals) > 0 else 1.0
    vmax = max(vmax, 1e-10)  # avoid zero range

    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        x=xlines if xlines is not None else None,
        y=inlines if inlines is not None else None,
        colorscale=SEISMIC_COLORSCALE,
        zmin=-vmax,
        zmax=vmax,
        colorbar=dict(title="Amplitude", tickfont=dict(color=PALETTE["text"])),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        xaxis_title="Crossline",
        yaxis_title="Inline",
        height=height,
        paper_bgcolor=PALETTE["bg_dark"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_inline_section(
    data: np.ndarray,
    il_idx: int,
    xlines: np.ndarray | None = None,
    time_axis: np.ndarray | None = None,
    title: str = "Inline Section",
    height: int = 500,
) -> go.Figure:
    """Plot an inline section (xline vs time)."""
    section = data[il_idx, :, :]  # (n_xl, n_samp)

    finite_vals = section[np.isfinite(section)]
    vmax = float(np.percentile(np.abs(finite_vals), 98)) if len(finite_vals) > 0 else 1.0
    vmax = max(vmax, 1e-10)

    fig = go.Figure(data=go.Heatmap(
        z=section.T,
        x=xlines if xlines is not None else None,
        y=time_axis if time_axis is not None else None,
        colorscale=SEISMIC_COLORSCALE,
        zmin=-vmax,
        zmax=vmax,
        colorbar=dict(title="Amp"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        xaxis_title="Crossline",
        yaxis_title="Time (ms)",
        height=height,
        paper_bgcolor=PALETTE["bg_dark"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_trace_comparison(
    trace_2d: np.ndarray,
    trace_3d: np.ndarray,
    trace_corr: np.ndarray | None = None,
    time_axis: np.ndarray | None = None,
    title: str = "Trace Comparison",
    height: int = 450,
) -> go.Figure:
    """Plot 2D vs 3D trace comparison, optionally with corrected trace."""
    t = time_axis if time_axis is not None else np.arange(len(trace_2d))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=trace_3d, name="3D Reference",
                             line=dict(color=PALETTE["success"], width=2)))
    fig.add_trace(go.Scatter(x=t, y=trace_2d, name="2D Original",
                             line=dict(color=PALETTE["danger"], width=1.5, dash="dash")))
    if trace_corr is not None:
        fig.add_trace(go.Scatter(x=t, y=trace_corr, name="2D Corrected",
                                 line=dict(color=PALETTE["primary"], width=2)))

    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude",
        height=height,
        paper_bgcolor=PALETTE["bg_dark"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_spectrum_comparison(
    freqs: np.ndarray,
    spec_2d: np.ndarray,
    spec_3d: np.ndarray,
    spec_corr: np.ndarray | None = None,
    title: str = "Amplitude Spectrum",
    height: int = 400,
) -> go.Figure:
    """Compare amplitude spectra of 2D / 3D / corrected."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=spec_3d, name="3D",
                             line=dict(color=PALETTE["success"], width=2)))
    fig.add_trace(go.Scatter(x=freqs, y=spec_2d, name="2D",
                             line=dict(color=PALETTE["danger"], width=1.5, dash="dash")))
    if spec_corr is not None:
        fig.add_trace(go.Scatter(x=freqs, y=spec_corr, name="Corrected",
                                 line=dict(color=PALETTE["primary"], width=2)))

    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=height,
        paper_bgcolor=PALETTE["bg_dark"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_map_with_lines(
    coords_3d: np.ndarray,
    lines_coords: list[np.ndarray],
    line_names: list[str],
    polygon_xy: np.ndarray | None = None,
    title: str = "Survey Map",
    height: int = 550,
) -> go.Figure:
    """Plot map view: 3D coverage, 2D lines, expansion polygon."""
    fig = go.Figure()

    # 3D coverage as scatter (subsample if large)
    n3 = len(coords_3d)
    step = max(1, n3 // 5000)
    fig.add_trace(go.Scattergl(
        x=coords_3d[::step, 0], y=coords_3d[::step, 1],
        mode="markers",
        marker=dict(size=2, color=PALETTE["text_muted"], opacity=0.3),
        name="3D Coverage",
    ))

    # 2D lines
    colours = px.colors.qualitative.Vivid
    for i, (lc, ln) in enumerate(zip(lines_coords, line_names)):
        fig.add_trace(go.Scattergl(
            x=lc[:, 0], y=lc[:, 1],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(color=colours[i % len(colours)], width=2),
            name=ln,
        ))

    # Polygon
    if polygon_xy is not None:
        fig.add_trace(go.Scatter(
            x=polygon_xy[:, 0], y=polygon_xy[:, 1],
            mode="lines",
            line=dict(color=PALETTE["warning"], width=2, dash="dot"),
            name="Expansion Polygon",
            fill="none",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        xaxis_title="X",
        yaxis_title="Y",
        height=height,
        paper_bgcolor=PALETTE["bg_dark"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(scaleanchor="y"),
    )
    return fig


def plot_metrics_radar(metrics: dict[str, float], title: str = "Quality Metrics", height: int = 400) -> go.Figure:
    """Radar / polar chart of normalised quality metrics."""
    labels = list(metrics.keys())
    values = list(metrics.values())
    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        line=dict(color=PALETTE["primary"]),
        fillcolor=f"rgba(99,102,241,0.2)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        polar=dict(
            bgcolor=PALETTE["bg_card"],
            radialaxis=dict(visible=True, range=[0, max(values) * 1.1], gridcolor="#313244"),
            angularaxis=dict(gridcolor="#313244"),
        ),
        paper_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        height=height,
        showlegend=False,
    )
    return fig


def plot_convergence(costs: list[float], title: str = "Convergence", height: int = 350) -> go.Figure:
    """Line plot of iteration cost for POCS / MSSA."""
    fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(costs) + 1)),
        y=costs,
        mode="lines+markers",
        line=dict(color=PALETTE["primary"], width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"])),
        xaxis_title="Iteration",
        yaxis_title="Cost",
        height=height,
        paper_bgcolor=PALETTE["bg_dark"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
    )
    return fig


def section_header(title: str, icon: str = "") -> None:
    """Render a styled section header."""
    st.markdown(
        f"""
        <div style="margin:24px 0 12px 0;">
            <span style="font-size:1.4rem; font-weight:700; color:{PALETTE['text']};">
                {icon}&ensp;{title}
            </span>
            <hr style="border:none; border-top:2px solid #313244; margin-top:8px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
