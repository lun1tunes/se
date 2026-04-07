"""Pydantic configuration models for seis2cube pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CoordSource(str, Enum):
    """Which trace-header fields to use for X/Y coordinates."""
    SOURCE = "source"       # SourceX / SourceY  (bytes 73-76 / 77-80)
    GROUP = "group"         # GroupX / GroupY     (bytes 81-84 / 85-88)
    CDP = "cdp"             # CDP_X / CDP_Y      (bytes 181-184 / 185-188)


class ComputeBackend(str, Enum):
    NUMPY = "numpy"
    DASK = "dask"


class CalibrationMethod(str, Enum):
    GLOBAL_SHIFT = "global_shift"
    WINDOWED = "windowed"
    LINEAR_REGRESSION = "linear_regression"
    GBDT = "gbdt"
    DL = "dl"


class InterpolationMethod(str, Enum):
    IDW = "idw"
    KRIGING = "kriging"
    POCS = "pocs"
    MSSA = "mssa"
    DL = "dl"


class POCSTransform(str, Enum):
    FFT = "fft"
    WAVELET = "wavelet"
    CURVELET = "curvelet"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class SegyHeaderBytes(BaseModel):
    """Byte positions (1-indexed as per SEG-Y spec) for key header words."""
    inline: int = Field(default=189, description="Byte position for inline number")
    xline: int = Field(default=193, description="Byte position for crossline number")
    coord_source: CoordSource = Field(default=CoordSource.CDP)
    coord_scalar_byte: int = Field(default=71, description="Byte for coordinate scalar")
    coord_units_byte: int = Field(default=89, description="Byte for coordinate units")


class ChunkingConfig(BaseModel):
    inline_chunk: int = Field(default=64)
    xline_chunk: int = Field(default=64)
    time_chunk: int = Field(default=256)
    format: Literal["zarr", "netcdf"] = "zarr"


class IOConfig(BaseModel):
    mmap: bool = Field(default=True, description="Use memory-mapping via segyio f.mmap()")
    ignore_geometry: bool = Field(default=False)
    strict: bool = Field(default=True)
    chunking: ChunkingConfig | None = None


class CRSConfig(BaseModel):
    source_crs: str | None = Field(default=None, description="EPSG code or WKT of input data CRS")
    target_crs: str = Field(default="EPSG:32637", description="Projected CRS for computations")


class CalibrationConfig(BaseModel):
    method: CalibrationMethod = CalibrationMethod.GLOBAL_SHIFT
    window_ms: float = Field(default=400.0, description="Window length in ms for windowed methods")
    window_overlap_ms: float = Field(default=100.0, description="Window overlap in ms")
    holdout_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    holdout_mode: Literal["segment", "line"] = "segment"
    spectral_matching: bool = Field(default=False)
    # Regression / GBDT
    features: list[str] = Field(
        default_factory=lambda: ["amplitude", "envelope", "rms", "d_dt"],
        description="Feature names for regression-based calibrators",
    )
    # GBDT specific
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1


class InterpolationConfig(BaseModel):
    method: InterpolationMethod = InterpolationMethod.POCS
    # IDW
    idw_power: float = 2.0
    idw_max_neighbours: int = 12
    # POCS
    pocs_niter: int = Field(default=100, description="Number of POCS iterations")
    pocs_transform: POCSTransform = POCSTransform.FFT
    pocs_fast: bool = Field(default=True, description="Use FPOCS (faster convergence)")
    pocs_threshold_schedule: Literal["linear", "exponential"] = "exponential"
    # MSSA
    mssa_rank: int = Field(default=20, description="Target rank for MSSA/SVD")
    mssa_window: int = Field(default=50, description="Hankel window length")
    # Simulation for hyperparameter selection inside 3D
    sim_n_trials: int = Field(default=10)


class ComputeConfig(BaseModel):
    backend: ComputeBackend = ComputeBackend.NUMPY
    n_workers: int = Field(default=4)
    memory_limit: str = Field(default="4GB")
    temp_dir: str | None = None


class BlendConfig(BaseModel):
    """Controls the seam blending between original 3D and extended region."""
    enabled: bool = True
    taper_width_traces: int = Field(default=10, description="Number of traces for cosine taper")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Top-level configuration for the seis2cube pipeline."""
    cube3d_path: Path
    lines2d_paths: list[Path]
    expand_polygon_path: Path
    out_cube_path: Path = Field(default=Path("output/extended_cube.segy"))
    qc_report_dir: Path = Field(default=Path("qc_report"))

    header_bytes: SegyHeaderBytes = Field(default_factory=SegyHeaderBytes)
    io: IOConfig = Field(default_factory=IOConfig)
    crs: CRSConfig = Field(default_factory=CRSConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    blend: BlendConfig = Field(default_factory=BlendConfig)

    # Grid extension
    grid_inline_step: float | None = Field(
        default=None, description="Override inline step for extended grid (meters)"
    )
    grid_xline_step: float | None = Field(
        default=None, description="Override crossline step for extended grid (meters)"
    )

    @model_validator(mode="after")
    def _ensure_paths(self) -> "PipelineConfig":
        if self.cube3d_path.suffix.lower() not in (".sgy", ".segy"):
            raise ValueError(f"cube3d_path must be a SEG-Y file, got {self.cube3d_path}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
