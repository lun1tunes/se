# seis2cube

**Extend 3D SEG-Y cubes from 2D seismic profiles with calibration against existing 3D.**

## Overview

`seis2cube` builds a new 3D post-stack SEG-Y volume that covers an expanded area by:

1. **Calibrating** 2D SEG-Y lines to match an existing 3D cube (amplitude, time shift, phase, spectrum) in their overlap zone.
2. **Interpolating** between calibrated 2D lines to fill a regular 3D grid in the expansion area.
3. **Assembling** the original 3D (as ground truth) and the reconstructed extension into a single output SEG-Y.

### Key features

- **SEG-Y Rev 1 compliance** — configurable header byte positions (inline/xline/X/Y), multiple coordinate sources (Source, Group, CDP), proper scalar handling.
- **Multi-level calibration** (Strategy pattern, switchable via config):
  - Level 0: Global Δt / gain / phase / spectral matching filter.
  - Level 1: Windowed (time-varying) shift & gain.
  - Level 2: Ridge / ElasticNet / Huber regression on trace features.
  - Level 3: Gradient Boosting (sklearn / XGBoost / LightGBM).
- **Multi-level interpolation**:
  - IDW per time-slice (baseline).
  - POCS / FPOCS with FFT or wavelet thresholding.
  - MSSA (Multichannel Singular Spectrum Analysis).
- **Honest validation** — calibration and interpolation are tuned/evaluated on hold-out data within the existing 3D (spatial splits only, no random-sample leakage).
- **Performance** — lazy segyio trace reading, optional `mmap()`, numpy.memmap / Zarr intermediate storage, Dask parallelism for time-slice processing.
- **CRS reprojection** via pyproj.
- **QC reporting** — JSON + CSV metrics, before/after calibration comparison.

## Installation

```bash
pip install -e .            # core dependencies
pip install -e ".[dask]"    # + Dask/xarray/Zarr/segysak
pip install -e ".[ml]"      # + XGBoost/LightGBM
pip install -e ".[all]"     # everything
```

## Quick start

```bash
# 1. Validate config & inputs
seis2cube validate --config config_example.yaml

# 2. Run the full pipeline
seis2cube run --config config_example.yaml -v
```

## Configuration

See [`config_example.yaml`](config_example.yaml) for a fully commented template.
Key sections:

| Section | Purpose |
|---------|---------|
| `header_bytes` | SEG-Y inline/xline/coord byte positions |
| `io` | mmap, strict, ignore_geometry, chunking |
| `crs` | Source & target CRS (EPSG codes) |
| `calibration` | Method, windows, holdout, features |
| `interpolation` | Method, POCS/MSSA/IDW parameters |
| `compute` | numpy vs dask backend, workers, memory |
| `blend` | Seam blending at 3D/extension boundary |

## Architecture

```
seis2cube/
├── config.py              # Pydantic config models
├── cli.py                 # Click CLI
├── io/
│   ├── segy_dataset.py    # SegyDataset (read abstraction over segyio)
│   ├── segy_writer.py     # SegyWriter3D (structured SEG-Y output)
│   └── chunked_storage.py # memmap / Zarr / Dask adapters
├── geometry/
│   ├── crs_converter.py   # CRS reprojection (pyproj)
│   ├── geometry_model.py  # AffineGridMapper / KDTreeMapper
│   └── overlap_detector.py# 2D↔3D overlap classification
├── models/
│   ├── line2d.py          # Line2D data model
│   └── volume.py          # TargetGrid, SparseVolume
├── calibration/
│   ├── base.py            # CalibrationStrategy ABC
│   ├── global_shift.py    # Level 0: Δt/gain/phase
│   ├── windowed.py        # Level 1: windowed corrections
│   ├── regression.py      # Level 2: linear regression
│   └── gbdt.py            # Level 3: gradient boosting
├── interpolation/
│   ├── base.py            # InterpolationStrategy ABC
│   ├── idw.py             # IDW per time-slice
│   ├── pocs.py            # POCS / FPOCS
│   └── mssa.py            # MSSA reconstruction
├── pipeline/
│   ├── volume_builder.py  # Grid construction, assembly, blending
│   └── runner.py          # Full pipeline orchestration
├── qc/
│   ├── metrics.py         # CalibrationMetrics, InterpolationMetrics
│   └── reporter.py        # QCReporter (JSON/CSV)
└── utils/
    ├── spectral.py        # Hilbert, phase rotation, matching filter, xcorr
    └── array_utils.py     # Windowing, blending, feature extraction
```

## Pipeline steps

1. **Ingest & Validate** — open 3D and 2D SEG-Y files, check dt/samples/format.
2. **CRS Conversion** — reproject all coordinates to a common projected CRS.
3. **Geometry Model** — build inline/xline ↔ (X,Y) mapping (affine or KDTree).
4. **Overlap Detection** — classify 2D trace points as inside/outside 3D coverage.
5. **Calibration Pairs** — match 2D amplitudes with bilinear-interpolated 3D amplitudes.
6. **Calibrate** — train selected strategy on overlap, evaluate on spatial holdout.
7. **Apply Calibration** — correct all 2D lines (including segments outside 3D).
8. **Target Grid** — define extended inline/xline grid covering 3D + expansion polygon.
9. **Tune Interpolation** — simulate sparse reconstruction inside 3D to select hyperparameters.
10. **Reconstruct Extension** — fill missing traces via selected interpolation method.
11. **Assemble** — merge original 3D + extension with optional cosine-taper blending.
12. **Write SEG-Y** — output structured 3D SEG-Y with correct headers.
13. **QC Report** — save metrics and diagnostics.

## References

- **SEG-Y Rev 1 standard** — binary header bytes 3217-3218 (sample interval µs), 3221-3222 (samples/trace), 3225-3226 (format code); trace header coordinate scalar bytes 71-72.
- **segyio** — `segyio.open(iline=, xline=, strict=, ignore_geometry=)`, `f.mmap()`, lazy trace interface.
- **POCS** — iterative thresholding reconstruction for irregularly sampled seismic data.
- **MSSA / I-MSSA** — low-rank Hankel-matrix reconstruction, handles off-grid observations.
- **Multi-vintage matching** — amplitude, time shift, phase, and spectral cross-equalization.

## License

MIT
