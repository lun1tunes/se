"""Session state management for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import streamlit as st


class PipelineStage(str, Enum):
    IDLE = "idle"
    INGESTING = "ingesting"
    CRS_CONVERSION = "crs_conversion"
    GEOMETRY = "geometry"
    OVERLAP = "overlap"
    LOADING_2D = "loading_2d"
    CALIBRATION = "calibration"
    APPLYING_CAL = "applying_calibration"
    BUILDING_GRID = "building_grid"
    TUNING_INTERP = "tuning_interpolation"
    RECONSTRUCTING = "reconstructing"
    ASSEMBLING = "assembling"
    WRITING = "writing"
    QC = "qc"
    DONE = "done"
    ERROR = "error"


STAGE_LABELS = {
    PipelineStage.IDLE: "Ready",
    PipelineStage.INGESTING: "Ingesting 3D cube...",
    PipelineStage.CRS_CONVERSION: "Converting coordinates...",
    PipelineStage.GEOMETRY: "Building geometry model...",
    PipelineStage.OVERLAP: "Detecting overlap zones...",
    PipelineStage.LOADING_2D: "Loading 2D profiles...",
    PipelineStage.CALIBRATION: "Running calibration...",
    PipelineStage.APPLYING_CAL: "Applying calibration...",
    PipelineStage.BUILDING_GRID: "Building target grid...",
    PipelineStage.TUNING_INTERP: "Tuning interpolation on 3D...",
    PipelineStage.RECONSTRUCTING: "Reconstructing extension...",
    PipelineStage.ASSEMBLING: "Assembling final cube...",
    PipelineStage.WRITING: "Writing output SEG-Y...",
    PipelineStage.QC: "Generating QC report...",
    PipelineStage.DONE: "Pipeline complete!",
    PipelineStage.ERROR: "Error occurred",
}

STAGE_ORDER = [s for s in PipelineStage if s not in (PipelineStage.IDLE, PipelineStage.DONE, PipelineStage.ERROR)]


def init_state() -> None:
    """Initialise session state defaults."""
    defaults = {
        "pipeline_stage": PipelineStage.IDLE,
        "pipeline_progress": 0.0,
        "pipeline_log": [],
        "config_dict": None,
        "config_obj": None,
        "meta_3d": None,
        "lines_info": [],
        "cal_metrics_before": None,
        "cal_metrics_after": None,
        "interp_sim_metrics": None,
        "cube_volume": None,
        "recon_volume": None,
        "final_volume": None,
        "orig_mask": None,
        "target_grid": None,
        "qc_data": None,
        "error_msg": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_log(msg: str) -> None:
    st.session_state.pipeline_log.append(msg)


def set_stage(stage: PipelineStage, progress: float | None = None) -> None:
    st.session_state.pipeline_stage = stage
    if progress is not None:
        st.session_state.pipeline_progress = progress
