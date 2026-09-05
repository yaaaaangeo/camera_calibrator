"""
tests/test_windshield_workspace_ui.py
==========================================

WindshieldWorkspace/VectorFieldChartWidget 배선 테스트. 계산 정확성은 이미
tests/test_windshield_baseline.py 등이 담당하므로, 여기서는:
  * Spherical/Residual Ray/Spline 라디오 버튼이 비활성화 상태인지
    (사용자 스펙 - Phase 2+는 "Coming soon"으로만 보여야 함)
  * back_requested 시그널이 실제로 발생하는지
  * VectorFieldChartWidget이 데이터 없이도 paintEvent에서 죽지 않는지
만 확인한다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication  # noqa: E402

from calibration.windshield.base import WindshieldModelType  # noqa: E402
from ui.windshield_vector_field_view import VectorFieldChartWidget  # noqa: E402
from ui.windshield_workspace import WindshieldWorkspace  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_baseline_spherical_residual_ray_enabled_spline_disabled(qapp):
    """Baseline(STEP 1)/Spherical(STEP 2)/Residual Ray(STEP 3-A)는 모두 실제로
    구현됐으므로 라디오가 활성화돼 있어야 하고, Spline(Phase 4)만 여전히
    "Coming soon"으로 비활성 상태여야 한다."""
    workspace = WindshieldWorkspace()
    buttons = {
        button.property("windshield_model"): button
        for button in workspace._model_button_group.buttons()
    }
    assert buttons[WindshieldModelType.BASELINE.value].isEnabled()
    assert buttons[WindshieldModelType.BASELINE.value].isChecked()
    assert buttons[WindshieldModelType.SPHERICAL.value].isEnabled()
    assert buttons[WindshieldModelType.RESIDUAL_RAY.value].isEnabled()
    assert not buttons[WindshieldModelType.SPLINE.value].isEnabled()


def test_back_requested_signal_fires(qapp):
    workspace = WindshieldWorkspace()
    received = []
    workspace.back_requested.connect(lambda: received.append(True))

    # header의 첫 번째 버튼("← Calibration Home")을 직접 클릭 대신 시그널 소스를
    # 찾기보다, 위젯 트리에서 버튼을 찾아 클릭한다.
    from PySide6.QtWidgets import QPushButton

    home_button = workspace.findChildren(QPushButton)[0]
    assert home_button.text().startswith("←")
    home_button.click()

    assert received == [True]


def test_vector_field_widget_handles_empty_data_without_crashing(qapp):
    widget = VectorFieldChartWidget()
    widget.resize(400, 300)
    widget.set_spatial_error_map(None)
    widget.repaint()  # paintEvent를 직접 호출해 예외 없이 끝나는지 확인


def _make_workspace_with_config(qapp):
    import numpy as np
    from calibration.types import CameraModelType
    from calibration.windshield.base import WindshieldConfig

    workspace = WindshieldWorkspace()
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    workspace._windshield_config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
    )
    return workspace


def test_residual_ray_advanced_group_visible_only_when_selected(qapp):
    workspace = _make_workspace_with_config(qapp)
    buttons = {
        button.property("windshield_model"): button
        for button in workspace._model_button_group.buttons()
    }
    buttons[WindshieldModelType.RESIDUAL_RAY.value].setChecked(True)
    assert workspace.residual_ray_advanced_group.isVisibleTo(workspace)
    buttons[WindshieldModelType.BASELINE.value].setChecked(True)
    assert not workspace.residual_ray_advanced_group.isVisibleTo(workspace)


def test_residual_ray_auto_mode_sets_auto_grid_hint(qapp):
    workspace = _make_workspace_with_config(qapp)
    workspace.grid_mode_auto_radio.setChecked(True)
    workspace._apply_residual_ray_advanced_settings()
    hint = workspace._windshield_config.residual_ray_hint
    assert hint["auto_grid"] == 1.0
    assert "grid_rows" not in hint
    assert "grid_cols" not in hint
    assert "lambda_mag" in hint and "lambda_smooth" in hint


def test_residual_ray_manual_mode_sets_grid_rows_cols_hint(qapp):
    workspace = _make_workspace_with_config(qapp)
    workspace.grid_mode_manual_radio.setChecked(True)
    workspace.grid_rows_spin.setValue(4)
    workspace.grid_cols_spin.setValue(6)
    workspace._apply_residual_ray_advanced_settings()
    hint = workspace._windshield_config.residual_ray_hint
    assert hint["auto_grid"] == 0.0
    assert hint["grid_rows"] == 4.0
    assert hint["grid_cols"] == 6.0


def test_residual_ray_lambda_spinboxes_propagate_to_hint(qapp):
    workspace = _make_workspace_with_config(qapp)
    workspace.lambda_mag_spin.setValue(0.005)
    workspace.lambda_smooth_spin.setValue(0.02)
    workspace._apply_residual_ray_advanced_settings()
    hint = workspace._windshield_config.residual_ray_hint
    assert hint["lambda_mag"] == pytest.approx(0.005)
    assert hint["lambda_smooth"] == pytest.approx(0.02)


def test_residual_ray_diagnostics_panel_visible_only_for_residual_ray_result(qapp):
    from calibration.windshield.base import WindshieldCalibrationResult

    workspace = _make_workspace_with_config(qapp)
    K, D = workspace._windshield_config.base_camera_matrix, workspace._windshield_config.base_distortion

    baseline_result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.BASELINE,
        base_model_name=workspace._windshield_config.base_model_name,
        base_camera_matrix=K, base_distortion=D, success=True,
    )
    workspace._display_result(baseline_result)
    assert not workspace.residual_ray_diagnostics_group.isVisibleTo(workspace)

    residual_result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        base_model_name=workspace._windshield_config.base_model_name,
        base_camera_matrix=K, base_distortion=D, success=True,
        fitted_params={"grid_rows": 3.0, "grid_cols": 4.0, "diag_selection_mode_is_auto": 1.0},
    )
    workspace._display_result(residual_result)
    assert workspace.residual_ray_diagnostics_group.isVisibleTo(workspace)
    assert workspace.diag_selected_grid_label.text() == "3 x 4"
    assert workspace.diag_selection_mode_label.text() == "AUTO"


def test_vector_field_widget_renders_populated_map_without_crashing(qapp):
    from calibration.types import SpatialErrorCell, SpatialErrorMap

    smap = SpatialErrorMap(
        cells=[
            SpatialErrorCell(row=0, col=0, num_points=5, rms=1.2, p95=2.0, mean_dx=0.5, mean_dy=-0.3, direction_deg=-30.0),
            SpatialErrorCell(row=0, col=1, num_points=0),
        ],
        rows=1,
        cols=2,
    )
    widget = VectorFieldChartWidget()
    widget.resize(400, 300)
    widget.set_spatial_error_map(smap)
    widget.repaint()
