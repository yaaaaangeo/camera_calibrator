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


def test_only_baseline_radio_is_enabled(qapp):
    workspace = WindshieldWorkspace()
    buttons = {
        button.property("windshield_model"): button
        for button in workspace._model_button_group.buttons()
    }
    assert buttons[WindshieldModelType.BASELINE.value].isEnabled()
    assert buttons[WindshieldModelType.BASELINE.value].isChecked()
    for model in (WindshieldModelType.SPHERICAL, WindshieldModelType.RESIDUAL_RAY, WindshieldModelType.SPLINE):
        assert not buttons[model.value].isEnabled()


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
