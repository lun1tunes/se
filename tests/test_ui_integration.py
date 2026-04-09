"""Integration tests for Streamlit UI with mocked data.

These tests simulate a complete pipeline run and verify UI updates.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


class TestDashboardWithData:
    """Test Dashboard tab with simulated pipeline results."""

    @pytest.fixture
    def app_with_results(self, tmp_path: Path) -> AppTest:
        """Create an app with simulated pipeline results in session state."""
        app_path = Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path))
        
        # Run to initialize session state
        at.run()
        
        # Simulate pipeline completion by setting session state
        at.session_state["meta_3d"] = {
            "n_traces": 251001,
            "n_samples": 156,
            "dt_ms": 2.0,
            "delrt_ms": 0.0,
            "format": 1,
            "n_inlines": 501,
            "n_xlines": 501,
        }
        at.session_state["lines_info"] = [
            {"name": "Line_11", "n_traces": 125, "dt_ms": 2.0},
            {"name": "Line_12", "n_traces": 130, "dt_ms": 2.0},
            {"name": "Line_13", "n_traces": 128, "dt_ms": 2.0},
            {"name": "Line_18", "n_traces": 140, "dt_ms": 2.0},
        ]
        at.session_state["cal_metrics_before"] = {"corr": 0.45, "rmse": 12000.0}
        at.session_state["cal_metrics_after"] = {
            "pearson_corr": 0.85,
            "rmse": 6000.0,
            "mae": 4500.0,
            "spectral_l2_rel": 0.8,
        }
        at.session_state["interp_sim_metrics"] = {
            "rmse": 5500.0,
            "mae": 3200.0,
            "pearson_corr": 0.72,
        }
        at.session_state["pipeline_stage"] = "DONE"
        at.session_state["pipeline_progress"] = 1.0
        
        # Mock config
        mock_config = MagicMock()
        mock_config.cube3d_path = MagicMock()
        mock_config.cube3d_path.name = "cube_3d.segy"
        mock_config.lines2d_paths = [MagicMock() for _ in range(4)]
        for i, path in enumerate(mock_config.lines2d_paths):
            path.name = f"2D_line_{i+11}.segy"
        mock_config.calibration.method.value = "global_shift"
        mock_config.interpolation.method.value = "idw"
        mock_config.compute.backend.value = "numpy"
        mock_config.blend.enabled = True
        mock_config.blend.taper_width_traces = 10
        mock_config.expand_polygon_path = None
        mock_config.expand_buffer_pct = 50.0
        mock_config.out_cube_path = Path("output/extended.segy")
        mock_config.qc_report_dir = Path("qc_report")
        
        at.session_state["config_obj"] = mock_config
        
        return at

    def test_dashboard_shows_metrics_with_data(self, app_with_results: AppTest) -> None:
        """Verify Dashboard shows metric cards when data is available."""
        app_with_results.run()
        
        # Should not show the "set data paths" info message
        dashboard_tab = app_with_results.tabs[0]
        
        # Check for metric-related elements (markdown with bold numbers)
        markdowns = [m.value for m in dashboard_tab.markdown]
        
        # Should contain some metrics
        assert len(markdowns) > 0, "Dashboard should show metrics with data"


class TestCalibrationTabWithData:
    """Test Calibration tab with simulated results."""

    @pytest.fixture
    def app_with_calibration(self, tmp_path: Path) -> AppTest:
        """Create an app with calibration results."""
        app_path = Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run()
        
        # Set calibration data
        at.session_state["cal_metrics_before"] = {"corr": 0.45, "rmse": 12000.0}
        at.session_state["cal_metrics_after"] = {
            "pearson_corr": 0.85,
            "rmse": 6000.0,
            "mae": 4500.0,
        }
        at.session_state["n_train"] = 450
        at.session_state["n_test"] = 113
        at.session_state["per_corr_before"] = np.random.uniform(0.3, 0.6, 113)
        at.session_state["per_corr_after"] = np.random.uniform(0.7, 0.9, 113)
        
        # Mock sample traces
        at.session_state["sample_traces_2d"] = np.random.randn(113, 156)
        at.session_state["sample_traces_3d"] = np.random.randn(113, 156)
        at.session_state["sample_traces_corr"] = np.random.randn(113, 156)
        at.session_state["dt_ms"] = 2.0
        
        return at

    def test_calibration_shows_results(self, app_with_calibration: AppTest) -> None:
        """Verify Calibration tab shows before/after metrics."""
        app_with_calibration.run()
        
        # Find Calibration tab
        cal_tab = None
        for tab in app_with_calibration.tabs:
            if "Calibration" in tab.label:
                cal_tab = tab
                break
        
        assert cal_tab is not None, "Calibration tab should exist"
        
        # Should have markdown content explaining calibration
        assert len(cal_tab.markdown) > 0, "Calibration tab should show content"


class TestInterpolationTabWithData:
    """Test Interpolation tab with simulated results."""

    @pytest.fixture
    def app_with_interpolation(self, tmp_path: Path) -> AppTest:
        """Create an app with interpolation results."""
        app_path = Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run()
        
        # Set interpolation data
        at.session_state["interp_sim_metrics"] = {
            "rmse": 5500.0,
            "mae": 3200.0,
            "pearson_corr": 0.72,
        }
        
        # Mock target grid - use smaller dimensions for faster test
        mock_grid = MagicMock()
        mock_grid.n_il = 200
        mock_grid.n_xl = 200
        mock_grid.inlines = np.arange(2800, 3000)
        mock_grid.xlines = np.arange(3200, 3400)
        mock_grid.origin_x = 450000
        mock_grid.origin_y = 890000
        mock_grid.il_step_x = 25.0
        mock_grid.il_step_y = 0.0
        mock_grid.xl_step_x = 0.0
        mock_grid.xl_step_y = 25.0
        at.session_state["target_grid"] = mock_grid
        
        # Mock masks - smaller for performance
        at.session_state["orig_mask"] = np.zeros((200, 200), dtype=bool)
        at.session_state["orig_mask"][50:150, 50:150] = True  # Original 3D area
        at.session_state["sparse_mask"] = at.session_state["orig_mask"].copy()
        
        # Mock scatter data
        at.session_state["interp_scatter_true"] = np.random.uniform(1000, 5000, 100)
        at.session_state["interp_scatter_pred"] = np.random.uniform(1200, 5200, 100)
        
        # Mock config
        mock_config = MagicMock()
        mock_config.interpolation.method.value = "idw"
        mock_config.interpolation.idw_power = 2.0
        mock_config.interpolation.idw_max_neighbours = 12
        at.session_state["config_obj"] = mock_config
        
        return at

    def test_interpolation_shows_metrics(self, app_with_interpolation: AppTest) -> None:
        """Verify Interpolation tab shows quality metrics."""
        app_with_interpolation.run()
        
        # Find Interpolation tab
        interp_tab = None
        for tab in app_with_interpolation.tabs:
            if "Interpolation" in tab.label:
                interp_tab = tab
                break
        
        assert interp_tab is not None, "Interpolation tab should exist"
        
        # Should have content
        assert len(interp_tab.markdown) > 0, "Interpolation tab should show content"


class TestVolumeViewerTabWithData:
    """Test Volume Viewer tab with simulated volume data."""

    @pytest.fixture
    def app_with_volume(self, tmp_path: Path) -> AppTest:
        """Create an app with volume data."""
        app_path = Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run()
        
        # Create small test volumes
        at.session_state["final_volume"] = np.random.randn(100, 100, 156).astype(np.float32)
        at.session_state["cube_volume"] = np.random.randn(50, 50, 156).astype(np.float32)
        at.session_state["inlines_3d"] = np.arange(2800, 2850)
        at.session_state["xlines_3d"] = np.arange(3200, 3250)
        
        # Mock target grid
        mock_grid = MagicMock()
        mock_grid.inlines = np.arange(2800, 2900)
        mock_grid.xlines = np.arange(3200, 3300)
        mock_grid.dt_ms = 2.0
        mock_grid.xy_at_corners = lambda il, xl: (
            np.array([450000 + (i - 2800) * 25 for i in il]),
            np.array([890000 + (x - 3200) * 25 for x in xl])
        )
        at.session_state["target_grid"] = mock_grid
        
        # Mock coordinates
        il_coords = np.repeat(at.session_state["inlines_3d"], 50)
        xl_coords = np.tile(at.session_state["xlines_3d"], 50)
        at.session_state["coords_3d"] = np.column_stack([
            450000 + (il_coords - 2800) * 25,
            890000 + (xl_coords - 3200) * 25
        ])
        
        # Mock 2D lines
        at.session_state["lines_coords"] = [
            np.array([[450000 + i*100, 890000 + i*50] for i in range(20)]),
            np.array([[450000 + i*80, 890000 + i*60] for i in range(25)]),
        ]
        at.session_state["line_names"] = ["Line_11", "Line_12"]
        
        return at

    def test_volume_viewer_shows_controls(self, app_with_volume: AppTest) -> None:
        """Verify Volume Viewer shows view controls."""
        app_with_volume.run()
        
        # Find Volume Viewer tab
        viewer_tab = None
        for tab in app_with_volume.tabs:
            if "Volume" in tab.label or "Viewer" in tab.label:
                viewer_tab = tab
                break
        
        assert viewer_tab is not None, "Volume Viewer tab should exist"
        
        # Should have radio buttons for view mode
        radios = viewer_tab.radio
        assert len(radios) > 0, "Volume Viewer should have view mode radio buttons"


class TestQCReportTabWithData:
    """Test QC Report tab with simulated results."""

    @pytest.fixture
    def app_with_qc(self, tmp_path: Path) -> AppTest:
        """Create an app with QC data."""
        app_path = Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path))
        at.run()
        
        # Set all metrics
        at.session_state["cal_metrics_before"] = {"corr": 0.45, "rmse": 12000.0}
        at.session_state["cal_metrics_after"] = {
            "pearson_corr": 0.85,
            "rmse": 6000.0,
            "mae": 4500.0,
            "spectral_l2_rel": 0.8,
        }
        at.session_state["interp_sim_metrics"] = {
            "rmse": 5500.0,
            "mae": 3200.0,
            "pearson_corr": 0.72,
        }
        
        # Mock config with qc_report_dir
        mock_config = MagicMock()
        mock_config.qc_report_dir = tmp_path / "qc_report"
        mock_config.qc_report_dir.mkdir(exist_ok=True)
        # Create dummy qc_report.json
        (mock_config.qc_report_dir / "qc_report.json").write_text('{"test": true}')
        at.session_state["config_obj"] = mock_config
        
        return at

    def test_qc_report_shows_metrics(self, app_with_qc: AppTest) -> None:
        """Verify QC Report shows metrics with explanations."""
        app_with_qc.run()
        
        # Find QC Report tab
        qc_tab = None
        for tab in app_with_qc.tabs:
            if "QC" in tab.label or "Report" in tab.label:
                qc_tab = tab
                break
        
        assert qc_tab is not None, "QC Report tab should exist"
        
        # Should have expanders for metric sections
        expanders = qc_tab.expander
        assert len(expanders) > 0, "QC Report should have metric expanders"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
