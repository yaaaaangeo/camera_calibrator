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

QtWidgets = pytest.importorskip("PySide6.QtWidgets", reason="PySide6.QtWidgets is not importable in this environment")
QApplication = QtWidgets.QApplication

from calibration.windshield.base import WindshieldModelType  # noqa: E402
from ui.windshield_vector_field_view import VectorFieldChartWidget  # noqa: E402
from ui.windshield_workspace import WindshieldWorkspace  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_all_windshield_models_are_enabled():
    """Baseline/Spherical/Residual Ray(Grid+RBF)/Spline(Phase 4) 전부 실제로
    구현됐으므로 더 이상 비활성화된("Coming soon") 모델이 없어야 한다."""
    workspace = WindshieldWorkspace()
    buttons = {
        button.property("windshield_model"): button
        for button in workspace._model_button_group.buttons()
    }
    assert buttons[WindshieldModelType.BASELINE.value].isEnabled()
    assert buttons[WindshieldModelType.BASELINE.value].isChecked()
    assert buttons[WindshieldModelType.SPHERICAL.value].isEnabled()
    assert buttons[WindshieldModelType.RESIDUAL_RAY.value].isEnabled()
    assert buttons[WindshieldModelType.SPLINE.value].isEnabled()


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


def test_residual_ray_neural_settings_visible_only_when_neural_method_selected(qapp):
    """사용자 스펙 5-C번 - Grid/RBF를 고르면 Neural Settings가 숨겨지고,
    Neural을 고르면 보여야 한다."""
    workspace = _make_workspace_with_config(qapp)
    buttons = {
        button.property("windshield_model"): button
        for button in workspace._model_button_group.buttons()
    }
    buttons[WindshieldModelType.RESIDUAL_RAY.value].setChecked(True)

    workspace.residual_ray_method_grid_radio.setChecked(True)
    assert not workspace.residual_neural_settings_group.isVisibleTo(workspace)

    workspace.residual_ray_method_rbf_radio.setChecked(True)
    assert not workspace.residual_neural_settings_group.isVisibleTo(workspace)

    workspace.residual_ray_method_neural_radio.setChecked(True)
    assert workspace.residual_neural_settings_group.isVisibleTo(workspace)
    assert not workspace.residual_grid_settings_group.isVisibleTo(workspace)
    assert not workspace.residual_rbf_settings_group.isVisibleTo(workspace)


def test_neural_method_and_hyperparameter_mapping_to_hint(qapp):
    """사용자 스펙 5-C번 - method=="neural" 매핑 + 모든 hyperparameter
    spinbox(max epochs/learning rate/weight decay/lambda mag/lambda
    smooth/patience/seed/batch size)가 정확히 hint에 반영되는지."""
    workspace = _make_workspace_with_config(qapp)
    workspace.residual_ray_method_neural_radio.setChecked(True)
    workspace.neural_epochs_spin.setValue(250)
    workspace.neural_lr_spin.setValue(0.002)
    workspace.neural_weight_decay_spin.setValue(0.0005)
    workspace.neural_lambda_mag_spin.setValue(0.03)
    workspace.neural_lambda_smooth_spin.setValue(0.04)
    workspace.neural_patience_spin.setValue(20)
    workspace.neural_seed_spin.setValue(7)
    workspace.neural_batch_size_spin.setValue(32)

    workspace._apply_residual_ray_advanced_settings()
    hint = workspace._windshield_config.residual_ray_hint

    assert hint["method"] == "neural"
    assert hint["neural_max_epochs"] == 250.0
    assert hint["neural_learning_rate"] == pytest.approx(0.002)
    assert hint["neural_weight_decay"] == pytest.approx(0.0005)
    assert hint["neural_lambda_mag"] == pytest.approx(0.03)
    assert hint["neural_lambda_smooth"] == pytest.approx(0.04)
    assert hint["neural_patience"] == 20.0
    assert hint["neural_seed"] == 7.0
    assert hint["neural_batch_size"] == 32.0


def test_neural_diagnostics_panel_shows_expected_fields(qapp):
    """사용자 스펙 5-C번 - 가짜 Neural 결과를 넣었을 때 Method/Architecture/
    Best Epoch/Train-Val Ray Loss/Seed Stability가 화면에 반영되는지."""
    from calibration.windshield.base import WindshieldCalibrationResult

    workspace = _make_workspace_with_config(qapp)
    K, D = workspace._windshield_config.base_camera_matrix, workspace._windshield_config.base_distortion

    neural_result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        base_model_name=workspace._windshield_config.base_model_name,
        base_camera_matrix=K, base_distortion=D, success=True,
        fitted_params={
            "residual_ray_method": 2.0,
            "neural_num_hidden_layers": 3.0,
            "neural_hidden_dim_0": 32.0, "neural_hidden_dim_1": 64.0, "neural_hidden_dim_2": 32.0,
            "neural_param_count": 4387.0,
            "neural_best_epoch": 83.0,
            "neural_best_train_ray_loss": 0.0012,
            "neural_best_val_ray_loss": 0.0015,
            "neural_best_train_total_loss": 0.002,
            "neural_best_val_total_loss": 0.0023,
            "neural_batch_size": 64.0,
            "diag_seed_stability_mean_deg": 0.013,
            "diag_seed_stability_p95_deg": 0.031,
            "diag_selection_mode_is_auto": 0.0,
        },
    )
    workspace._display_result(neural_result)

    assert workspace.residual_ray_diagnostics_group.isVisibleTo(workspace)
    assert workspace.diag_residual_method_label.text() == "Neural"
    assert "32" in workspace.diag_neural_architecture_label.text()
    assert "64" in workspace.diag_neural_architecture_label.text()
    assert "83" in workspace.diag_neural_training_label.text()
    assert "64" in workspace.diag_neural_training_label.text()  # batch size
    assert workspace.diag_neural_total_loss_label.text() != "N/A"
    assert workspace.diag_neural_seed_stability_label.text() != "N/A"
    assert workspace.diag_selection_mode_label.text() == "Manual"


def test_spline_advanced_group_visible_only_when_selected(qapp):
    workspace = _make_workspace_with_config(qapp)
    buttons = {
        button.property("windshield_model"): button
        for button in workspace._model_button_group.buttons()
    }
    buttons[WindshieldModelType.SPLINE.value].setChecked(True)
    assert workspace.spline_advanced_group.isVisibleTo(workspace)
    buttons[WindshieldModelType.BASELINE.value].setChecked(True)
    assert not workspace.spline_advanced_group.isVisibleTo(workspace)


def test_spline_auto_mode_sets_auto_spline_hint(qapp):
    workspace = _make_workspace_with_config(qapp)
    workspace.spline_mode_auto_radio.setChecked(True)
    workspace._apply_spline_advanced_settings()
    hint = workspace._windshield_config.spline_hint
    assert hint["auto_spline"] == 1.0
    assert "spline_rows" not in hint
    assert "spline_cols" not in hint
    assert "lambda_mag" in hint and "lambda_smooth" in hint and "lambda_curve" in hint
    assert "max_displacement_m" in hint


def test_spline_manual_mode_sets_rows_cols_hint(qapp):
    workspace = _make_workspace_with_config(qapp)
    workspace.spline_mode_manual_radio.setChecked(True)
    workspace.spline_rows_spin.setValue(4)
    workspace.spline_cols_spin.setValue(6)
    workspace._apply_spline_advanced_settings()
    hint = workspace._windshield_config.spline_hint
    assert hint["auto_spline"] == 0.0
    assert hint["spline_rows"] == 4.0
    assert hint["spline_cols"] == 6.0


def test_spline_max_displacement_spinbox_converts_mm_to_meters(qapp):
    workspace = _make_workspace_with_config(qapp)
    workspace.spline_max_displacement_spin.setValue(15.0)  # mm
    workspace._apply_spline_advanced_settings()
    hint = workspace._windshield_config.spline_hint
    assert hint["max_displacement_m"] == pytest.approx(0.015)


def test_spline_diagnostics_panel_visible_only_for_spline_result(qapp):
    from calibration.windshield.base import WindshieldCalibrationResult

    workspace = _make_workspace_with_config(qapp)
    K, D = workspace._windshield_config.base_camera_matrix, workspace._windshield_config.base_distortion

    baseline_result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.BASELINE,
        base_model_name=workspace._windshield_config.base_model_name,
        base_camera_matrix=K, base_distortion=D, success=True,
    )
    workspace._display_result(baseline_result)
    assert not workspace.spline_diagnostics_group.isVisibleTo(workspace)

    spline_result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPLINE,
        base_model_name=workspace._windshield_config.base_model_name,
        base_camera_matrix=K, base_distortion=D, success=True,
        fitted_params={
            "sphere_radius": 5.0, "sphere_center_x": 0.0, "sphere_center_y": 0.0, "sphere_center_z": -4.9,
            "spline_rows": 3.0, "spline_cols": 4.0, "diag_selection_mode_is_auto": 1.0,
            "runtime_param_count": 12.0,
        },
    )
    workspace._display_result(spline_result)
    assert workspace.spline_diagnostics_group.isVisibleTo(workspace)
    assert workspace.diag_spline_grid_label.text() == "3 x 4"
    assert workspace.diag_spline_selection_mode_label.text() == "AUTO"


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
