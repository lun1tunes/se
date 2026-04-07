"""CLI entry point for seis2cube.

Usage::

    seis2cube run --config config.yaml
    seis2cube validate --config config.yaml
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger


@click.group()
@click.version_option(package_name="seis2cube")
def main() -> None:
    """seis2cube — Extend 3D SEG-Y cubes from 2D profiles."""
    pass


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to pipeline YAML config.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def run(config: Path, verbose: bool) -> None:
    """Execute the full seis2cube pipeline."""
    import sys
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    from seis2cube.config import PipelineConfig
    from seis2cube.pipeline.runner import PipelineRunner

    cfg = PipelineConfig.from_yaml(config)
    runner = PipelineRunner(cfg)
    out = runner.run()
    click.echo(f"Output SEG-Y: {out}")


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to pipeline YAML config.",
)
def validate(config: Path) -> None:
    """Validate config and check input files exist."""
    from seis2cube.config import PipelineConfig

    try:
        cfg = PipelineConfig.from_yaml(config)
    except Exception as e:
        click.echo(f"Config validation FAILED: {e}", err=True)
        raise SystemExit(1)

    errors = []
    if not cfg.cube3d_path.exists():
        errors.append(f"3D cube not found: {cfg.cube3d_path}")
    for lp in cfg.lines2d_paths:
        if not lp.exists():
            errors.append(f"2D line not found: {lp}")
    if not cfg.expand_polygon_path.exists():
        errors.append(f"Polygon not found: {cfg.expand_polygon_path}")

    if errors:
        for e in errors:
            click.echo(f"  ERROR: {e}", err=True)
        raise SystemExit(1)

    click.echo("Config valid. All input files found.")
    click.echo(f"  3D cube: {cfg.cube3d_path}")
    click.echo(f"  2D lines: {len(cfg.lines2d_paths)}")
    click.echo(f"  Polygon: {cfg.expand_polygon_path}")
    click.echo(f"  Calibration: {cfg.calibration.method.value}")
    click.echo(f"  Interpolation: {cfg.interpolation.method.value}")


if __name__ == "__main__":
    main()
