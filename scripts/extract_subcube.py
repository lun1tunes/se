#!/usr/bin/env python3
"""Extract a sub-cube from the full 3D SEG-Y to create smaller test data.

Reads inline-by-inline via segyio mmap (never loads full volume)
and writes only the IL/XL sub-range to a new SEG-Y file.

Usage:
    python scripts/extract_subcube.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import segyio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seis2cube.config import PipelineConfig

# ── Configuration ────────────────────────────────────────────────────
CFG_PATH = Path("configs/test_real_data.yaml")
OUT_DIR = Path("test_data2")

# Sub-region in IL/XL coordinates (inclusive)
IL_MIN, IL_MAX = 2800, 3300
XL_MIN, XL_MAX = 3200, 3700

# ── Load original config to get paths + header bytes ─────────────────
cfg = PipelineConfig.from_yaml(CFG_PATH)
src_path = cfg.cube3d_path
hb_il = cfg.header_bytes.inline   # 189
hb_xl = cfg.header_bytes.xline    # 193

OUT_DIR.mkdir(parents=True, exist_ok=True)
dst_path = OUT_DIR / "subcube_3d.segy"

print(f"Source:  {src_path}  ({src_path.stat().st_size / 1e9:.2f} GB)")
print(f"Target:  {dst_path}")
print(f"Region:  IL [{IL_MIN}..{IL_MAX}], XL [{XL_MIN}..{XL_MAX}]")

# ── Open source ──────────────────────────────────────────────────────
t0 = time.perf_counter()
src = segyio.open(str(src_path), "r", iline=hb_il, xline=hb_xl, strict=True)
src.mmap()

orig_inlines = src.ilines
orig_xlines = src.xlines
n_samples = len(src.samples)
dt_us = int(src.bin[segyio.BinField.Interval])

# Filter IL/XL
sub_il = np.array([il for il in orig_inlines if IL_MIN <= il <= IL_MAX])
sub_xl = np.array([xl for xl in orig_xlines if XL_MIN <= xl <= XL_MAX])

print(f"Original: {len(orig_inlines)} IL × {len(orig_xlines)} XL × {n_samples} samp")
print(f"Sub-cube: {len(sub_il)} IL × {len(sub_xl)} XL × {n_samples} samp")
n_traces = len(sub_il) * len(sub_xl)
est_gb = n_traces * (n_samples * 4 + 240) / 1e9
print(f"Output:   {n_traces:,} traces, ~{est_gb:.2f} GB")

# ── Create output SEG-Y ─────────────────────────────────────────────
spec = segyio.spec()
spec.sorting = 2  # inline sorting
spec.format = 1   # IBM float (or same as source)
spec.iline = hb_il
spec.xline = hb_xl
spec.samples = src.samples
spec.ilines = sub_il
spec.xlines = sub_xl

with segyio.create(str(dst_path), spec) as dst:
    # Copy binary header
    dst.bin = src.bin
    dst.bin[segyio.BinField.Interval] = dt_us

    trace_idx = 0
    total = len(sub_il)
    for i, il in enumerate(sub_il):
        # Read full inline from source (n_orig_xl, n_samples)
        il_data = src.iline[il]

        # Get headers for this inline to copy coordinate info
        # Find trace indices for this inline in source
        for j, xl in enumerate(sub_xl):
            # Find xl position in original xlines
            xl_pos = np.searchsorted(orig_xlines, xl)
            if xl_pos >= len(orig_xlines) or orig_xlines[xl_pos] != xl:
                # Should not happen if sub_xl is subset
                dst.trace[trace_idx] = np.zeros(n_samples, dtype=np.float32)
                trace_idx += 1
                continue

            # Write trace data
            dst.trace[trace_idx] = il_data[xl_pos].astype(np.float32)

            # Copy trace header from source
            src_trace_idx = np.searchsorted(orig_inlines, il) * len(orig_xlines) + xl_pos
            src_hdr = src.header[src_trace_idx]
            h = dst.header[trace_idx]
            # Copy key fields
            for field in [
                segyio.TraceField.INLINE_3D,
                segyio.TraceField.CROSSLINE_3D,
                segyio.TraceField.CDP_X,
                segyio.TraceField.CDP_Y,
                segyio.TraceField.SourceX,
                segyio.TraceField.SourceY,
                segyio.TraceField.GroupX,
                segyio.TraceField.GroupY,
                segyio.TraceField.SourceGroupScalar,
                segyio.TraceField.DelayRecordingTime,
                segyio.TraceField.ElevationScalar,
            ]:
                h[field] = src_hdr[field]

            trace_idx += 1

        if (i + 1) % 100 == 0 or i + 1 == total:
            elapsed = time.perf_counter() - t0
            pct = (i + 1) / total * 100
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{pct:5.1f}%] {i+1}/{total} inlines, "
                  f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s")

src.close()
elapsed = time.perf_counter() - t0

file_size_gb = dst_path.stat().st_size / 1e9
print(f"\n✅ Sub-cube saved: {dst_path}")
print(f"   Size: {file_size_gb:.2f} GB")
print(f"   Time: {elapsed:.1f}s")
print(f"   {len(sub_il)} IL × {len(sub_xl)} XL × {n_samples} samp")
