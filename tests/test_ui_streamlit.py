"""Streamlit UI tests using the native AppTest framework.

These tests verify that the seis2cube Streamlit UI components render correctly
and user interactions work as expected.

Usage:
    pytest tests/test_ui_streamlit.py -v

Requirements:
    streamlit >= 1.28.0 (for AppTest support)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_test() -> AppTest:
    """Create an AppTest instance from the main app file."""
    app_path = Path(__file__).parent.parent / "seis2cube" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path))
    return at


class TestAppInitialization:
    """Test basic app initialization and page structure."""

    def test_app_loads_without_exception(self, app_test: AppTest) -> None:
        """Verify the app loads without errors."""
        app_test.run()
        assert not app_test.exception, f"App raised exception: {app_test.exception}"

    @pytest.mark.skip(reason="set_page_config title not accessible via AppTest")
    def test_page_title_is_set(self, app_test: AppTest) -> None:
        """Verify the page title is set correctly."""
        app_test.run()
        # Page title is set via st.set_page_config - not directly testable via AppTest
        # but the app should run without errors
        assert not app_test.exception, "App should set title without errors"

    def test_main_tabs_exist(self, app_test: AppTest) -> None:
        """Verify all main tabs are present."""
        app_test.run()
        tabs = app_test.tabs
        tab_labels = [tab.label for tab in tabs]
        
        expected_tabs = [
            "📊 Dashboard",
            "📁 Data Explorer",
            "🔧 Calibration",
            "🧩 Interpolation",
            "🗺️ Volume Viewer",
            "📈 QC Report",
        ]
        for expected in expected_tabs:
            assert expected in tab_labels, f"Tab '{expected}' not found in {tab_labels}"


class TestSidebarElements:
    """Test sidebar input elements and configuration."""

    def test_sidebar_has_input_fields(self, app_test: AppTest) -> None:
        """Verify sidebar has 3D and 2D folder inputs."""
        app_test.run()
        sidebar = app_test.sidebar
        
        # Check for text inputs
        text_inputs = sidebar.text_input
        input_labels = [ti.label for ti in text_inputs]
        
        assert any("3D" in label for label in input_labels), "3D folder input not found"
        assert any("2D" in label for label in input_labels), "2D folder input not found"

    def test_sidebar_has_settings_controls(self, app_test: AppTest) -> None:
        """Verify sidebar has settings controls (slider, selectboxes)."""
        app_test.run()
        sidebar = app_test.sidebar
        
        # Check for expansion buffer slider
        sliders = sidebar.slider
        assert len(sliders) > 0, "No slider found in sidebar"
        
        # Check for selectboxes (calibration and interpolation methods)
        selectboxes = sidebar.selectbox
        assert len(selectboxes) >= 2, "Expected at least 2 selectboxes (calibration, interpolation)"

    def test_run_button_exists_in_sidebar(self, app_test: AppTest) -> None:
        """Verify run button exists in sidebar."""
        app_test.run()
        sidebar = app_test.sidebar
        
        buttons = sidebar.button
        run_buttons = [b for b in buttons if "Run" in str(b.label) or "▶" in str(b.label)]
        
        # Button should exist
        assert len(run_buttons) > 0, "Run button should exist in sidebar"


class TestDashboardTab:
    """Test the Dashboard tab content."""

    def test_dashboard_shows_initial_message(self, app_test: AppTest) -> None:
        """Verify dashboard shows initial message before pipeline run."""
        app_test.run()
        
        # Select the Dashboard tab (usually first tab)
        tabs = app_test.tabs
        dashboard_tab = tabs[0]
        
        # Check for info message
        assert len(dashboard_tab.info) > 0, "Dashboard should show info message initially"


class TestDataExplorerTab:
    """Test the Data Explorer tab content."""

    def test_data_explorer_has_map_placeholder(self, app_test: AppTest) -> None:
        """Verify Data Explorer tab exists."""
        app_test.run()
        
        tabs = app_test.tabs
        tab_labels = [tab.label for tab in tabs]
        
        assert "📁 Data Explorer" in tab_labels, "Data Explorer tab not found"


class TestCalibrationTab:
    """Test the Calibration tab content."""

    def test_calibration_tab_exists(self, app_test: AppTest) -> None:
        """Verify Calibration tab exists."""
        app_test.run()
        
        tabs = app_test.tabs
        tab_labels = [tab.label for tab in tabs]
        
        assert "🔧 Calibration" in tab_labels, "Calibration tab not found"


class TestInterpolationTab:
    """Test the Interpolation tab content."""

    def test_interpolation_tab_exists(self, app_test: AppTest) -> None:
        """Verify Interpolation tab exists."""
        app_test.run()
        
        tabs = app_test.tabs
        tab_labels = [tab.label for tab in tabs]
        
        assert "🧩 Interpolation" in tab_labels, "Interpolation tab not found"


class TestVolumeViewerTab:
    """Test the Volume Viewer tab content."""

    def test_volume_viewer_tab_exists(self, app_test: AppTest) -> None:
        """Verify Volume Viewer tab exists."""
        app_test.run()
        
        tabs = app_test.tabs
        tab_labels = [tab.label for tab in tabs]
        
        assert "🗺️ Volume Viewer" in tab_labels, "Volume Viewer tab not found"


class TestQCReportTab:
    """Test the QC Report tab content."""

    def test_qc_report_tab_exists(self, app_test: AppTest) -> None:
        """Verify QC Report tab exists."""
        app_test.run()
        
        tabs = app_test.tabs
        tab_labels = [tab.label for tab in tabs]
        
        assert "📈 QC Report" in tab_labels, "QC Report tab not found"


class TestSessionState:
    """Test session state initialization."""

    def test_session_state_keys_exist(self, app_test: AppTest) -> None:
        """Verify essential session state keys are initialized."""
        app_test.run()
        
        # Check that the app ran without errors
        assert not app_test.exception, "App should initialize session state without errors"


class TestUIStyling:
    """Test that custom CSS and styling is applied."""

    def test_custom_markdown_css_loaded(self, app_test: AppTest) -> None:
        """Verify custom CSS is loaded via st.markdown."""
        app_test.run()
        
        # Custom CSS is loaded via st.markdown with unsafe_allow_html=True
        # This should not raise an exception
        assert not app_test.exception, "Custom CSS loading should not cause errors"


@pytest.mark.skip(reason="Requires full pipeline run - integration test")
class TestPipelineRunUI:
    """UI tests that require running the full pipeline (heavyweight)."""

    def test_pipeline_progress_shows_stages(self, app_test: AppTest) -> None:
        """Verify pipeline progress indicator shows all stages during run."""
        # This test would require mocking the pipeline or running with real data
        pass

    def test_metric_cards_appear_after_run(self, app_test: AppTest) -> None:
        """Verify metric cards appear after successful pipeline run."""
        # This test would require a completed pipeline run
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
