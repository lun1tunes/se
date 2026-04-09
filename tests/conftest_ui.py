"""Shared fixtures for Streamlit UI tests.

Import this in test files or add to conftest.py
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="session")
def app_path() -> Path:
    """Return the path to the main Streamlit app."""
    return Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"


@pytest.fixture
def fresh_app(app_path: Path) -> AppTest:
    """Create a fresh AppTest instance with clean session state."""
    at = AppTest.from_file(str(app_path))
    return at


@pytest.fixture
def app_with_mock_config(fresh_app: AppTest) -> AppTest:
    """Create an app with a mock config loaded."""
    # Mock the config_obj in session state
    mock_config = MagicMock()
    mock_config.cube3d_path = MagicMock()
    mock_config.cube3d_path.name = "test_3d.segy"
    mock_config.lines2d_paths = [MagicMock()]
    mock_config.lines2d_paths[0].name = "test_2d.segy"
    mock_config.expand_polygon_path = None
    mock_config.expand_buffer_pct = 50.0
    mock_config.calibration.method.value = "global_shift"
    mock_config.interpolation.method.value = "idw"
    mock_config.compute.backend.value = "numpy"
    mock_config.blend.enabled = True
    mock_config.blend.taper_width_traces = 10
    mock_config.out_cube_path = Path("output/test.segy")
    mock_config.qc_report_dir = Path("qc_report")
    
    # Mock the interpolation config
    mock_config.interpolation.idw_power = 2.0
    mock_config.interpolation.idw_max_neighbours = 12
    mock_config.interpolation.pocs_niter = 100
    mock_config.interpolation.mssa_rank = 20
    
    # Set in session state (this requires running the app first)
    fresh_app.run()
    fresh_app.session_state["config_obj"] = mock_config
    
    return fresh_app


@pytest.fixture
def mock_pipeline_runner():
    """Provide a mock pipeline runner for testing UI without running real pipeline."""
    with patch("seis2cube.ui.app._run_pipeline") as mock_run:
        mock_run.return_value = None
        yield mock_run
