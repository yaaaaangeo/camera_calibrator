"""
tests/test_radial_profile_view.py
======================================

Edge Error Map(⑤ 탭)의 y축 스케일 회귀 테스트.

실제 사용자 버그: RadialProfileChartWidget이 "지금 선택된 모델 자신의
최댓값"으로 매번 y축을 다시 스케일해서, 모델(Ideal Pinhole/Brown-Conrady/
Rational/Fisheye)을 콤보박스로 오갈 때마다 축이 바뀌었다 - 그러면 막대
높이만 봐서는 어느 모델이 실제로 더 나은지 비교가 안 된다("세로 축이 계속
바뀌니깐 뭐가 더 좋은지 잘 모르겠어"라는 피드백).

고친 동작: RadialProfileView.set_results()가 전달된 모델 전체를 통틀어
가장 큰 구간 평균 오차 하나를 구해서 고정 스케일로 쓰고, 모델을 바꿔도 이
스케일은 그대로 유지되어야 한다.
"""

from __future__ import annotations

import pytest

from calibration.types import (
    CalibrationResult,
    CameraModelType,
    RadialBin,
    RadialErrorProfile,
)

# 정합성 마감 라운드 - 최상위 `PySide6` 패키지 import는 성공해도 실제
# 컴파일된 확장(QtWidgets.pyd 등) 로드는 서브모듈 import 시점에 비로소
# 일어난다. 이 skip을 "PySide6.QtWidgets"까지 명시해야, 그 로드가 실패하는
# 환경(예: DLL 로드 실패)에서 collection error 대신 정상적으로 skip된다.
pytest.importorskip("PySide6.QtWidgets", reason="PySide6.QtWidgets is not importable in this environment")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.radial_profile_view import RadialProfileChartWidget, RadialProfileView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _profile(errors: list[float]) -> RadialErrorProfile:
    bins = [
        RadialBin(radius_min=i * 100.0, radius_max=(i + 1) * 100.0, mean_error=e, num_points=10)
        for i, e in enumerate(errors)
    ]
    return RadialErrorProfile(bins=bins)


def _result(model: CameraModelType, errors: list[float]) -> CalibrationResult:
    return CalibrationResult(
        model_name=model, rms_error=1.0, success=True, radial_profile=_profile(errors)
    )


def test_global_max_is_the_largest_error_across_all_models(qapp):
    """Fisheye가 압도적으로 오차가 큰 상황 - 고정 스케일은 Fisheye 기준이어야 한다."""
    results = {
        CameraModelType.PINHOLE: _result(CameraModelType.PINHOLE, [0.5, 1.0, 2.0]),
        CameraModelType.EXTENDED_PINHOLE: _result(CameraModelType.EXTENDED_PINHOLE, [0.3, 0.6, 0.9]),
        CameraModelType.FISHEYE: _result(CameraModelType.FISHEYE, [1.0, 3.0, 8.0]),
    }
    view = RadialProfileView()
    view.set_results(results)

    assert view._global_max_error == pytest.approx(8.0 * 1.15)


def test_switching_selected_model_does_not_change_the_scale(qapp):
    """콤보박스로 모델을 바꿔도(= _refresh_chart가 다시 호출돼도) 차트에
    전달되는 고정 스케일 값 자체는 그대로여야 한다 - 이게 이 버그의 핵심."""
    results = {
        CameraModelType.PINHOLE: _result(CameraModelType.PINHOLE, [0.5, 1.0, 2.0]),
        CameraModelType.EXTENDED_PINHOLE: _result(CameraModelType.EXTENDED_PINHOLE, [0.3, 0.6, 0.9]),
        CameraModelType.FISHEYE: _result(CameraModelType.FISHEYE, [1.0, 3.0, 8.0]),
    }
    view = RadialProfileView()
    view.set_results(results)
    fixed_scale_before = view.chart._fixed_max_error

    view.select_model(CameraModelType.EXTENDED_PINHOLE)  # 오차가 훨씬 작은 모델로 전환
    fixed_scale_after_switch = view.chart._fixed_max_error

    assert fixed_scale_before == fixed_scale_after_switch == pytest.approx(8.0 * 1.15)


def test_chart_widget_uses_fixed_max_when_provided(qapp):
    """차트 위젯 자체(순수 페인팅 로직) 단위 테스트 - fixed_max_error를 주면
    그 프로필 자신의 최댓값이 아니라 주어진 값을 y축 상한으로 써야 한다.

    qapp을 명시적으로 받는다 - QApplication이 아직 없는 상태에서 QWidget
    (RadialProfileChartWidget)을 만들면 플랫폼에 따라 Fatal Python
    error(SIGABRT)로 프로세스 자체가 죽을 수 있다(실측:
    tests/test_windshield_workspace_ui.py에서 동일 패턴을 Linux+offscreen
    platform으로 재현/확인). 지금은 이 파일의 앞선 테스트가 이미
    QApplication을 만들어놔서 우연히 안전하지만, 이 테스트만 따로
    실행하거나 테스트 순서가 바뀌면 재발할 수 있다."""
    chart = RadialProfileChartWidget()
    small_profile = _profile([0.1, 0.2])

    chart.set_profile(small_profile, fixed_max_error=100.0)
    assert chart._fixed_max_error == 100.0
    assert chart._profile is small_profile


def test_chart_widget_falls_back_to_own_scale_without_fixed_max(qapp):
    """fixed_max_error를 안 주면(예: 결과가 하나뿐일 때) 기존 동작 그대로
    자기 자신의 최댓값을 쓴다 - 하위 호환 확인. qapp을 받는 이유는 위
    test_chart_widget_uses_fixed_max_when_provided 참고."""
    chart = RadialProfileChartWidget()
    profile = _profile([0.1, 0.2])

    chart.set_profile(profile)
    assert chart._fixed_max_error is None


def test_no_successful_results_leaves_global_max_none(qapp):
    view = RadialProfileView()
    view.set_results({CameraModelType.PINHOLE: CalibrationResult(
        model_name=CameraModelType.PINHOLE, rms_error=0.0, success=False,
    )})
    assert view._global_max_error is None
