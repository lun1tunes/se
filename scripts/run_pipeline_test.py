#!/usr/bin/env python3
"""Quick pipeline test with auto-expand polygon + memory safety."""
import sys, resource, time
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {message}")

from seis2cube.config import PipelineConfig
from seis2cube.pipeline.runner import PipelineRunner

cfg = PipelineConfig(
    cube3d_path="test_data2/subcube_3d.segy",
    lines2d_paths=[
        "test_data2/2Д_профиль11",
        "test_data2/2Д_профиль12",
        "test_data2/2Д_профиль13",
        "test_data2/2Д_профиль18",
    ],
    expand_buffer_pct=50.0,
)
logger.info("expand={}%  max_grid={}GB", cfg.expand_buffer_pct, cfg.max_grid_memory_gb)

t0 = time.time()
runner = PipelineRunner(cfg)
out = runner.run()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
logger.info("DONE in {:.1f}s  Peak RSS: {:.0f} MB  Output: {}", time.time() - t0, rss, out)
