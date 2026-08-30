"""
tests/test_calibration_accuracy.py
======================================

이전까지 테스트 스위트에 없었던 갭을 메우는 테스트: "계산이 성공했는가"가
아니라 "복원된 파라미터가 알려진 정답에 실제로 가까운가"를 검증한다.

calibration/self_check.py에 있는 합성 데이터 생성 + 비교 로직을 그대로
재사용한다 (GUI의 "자체 진단" 버튼도 같은 함수를 쓴다 - 로직 중복 금지).

이 테스트가 실패한다면 십중팔구 계산 로직(calibrate_pinhole,
calibrate_extended_pinhole, 또는 OpenCV 버전업으로 인한 API 변화) 어딘가가
진짜로 깨진 것이다 - 임계값은 여러 시드에서 실측한 값에 여유를 두고 잡았다
(calibration/self_check.py 모듈 docstring 참고).
"""

from __future__ import annotations

import pytest

from calibration.self_check import (
    run_pinhole_accuracy_check,
    run_extended_pinhole_accuracy_check,
)

# 합성 이미지 렌더링 + 여러 시드 x 여러 모델 조합이라 다른 테스트보다 느리다
# (전체 실행 ~1~2분). 빠른 배선 확인만 필요하면 pytest -m "not slow"로 건너뛸 것.
pytestmark = pytest.mark.slow


# 여러 시드에서 안정적으로 통과하는지 확인 (특정 시드에서만 우연히 통과하는
# 걸 방지) - 대화 중 7개 시드로 실측 검증한 값 중 대표로 3개를 골랐다.
SEEDS = [3, 7, 42]


class TestGroundTruthAccuracy:
    """(1) 3D 회전이 충분한 잘 조건화된 합성 데이터로 fx/fy/distortion이
    정답의 임계값 이내인지 검증."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extended_pinhole_recovers_focal_length(self, seed):
        result = run_extended_pinhole_accuracy_check(use_rational_model=False, seed=seed)
        assert result.success, result.message
        assert result.fx_error_pct is not None and result.fx_error_pct <= 6.0, (
            f"fx 오차가 너무 큽니다: {result.message}"
        )
        assert result.fy_error_pct is not None and result.fy_error_pct <= 6.0, (
            f"fy 오차가 너무 큽니다: {result.message}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extended_pinhole_recovers_principal_point(self, seed):
        result = run_extended_pinhole_accuracy_check(use_rational_model=False, seed=seed)
        assert result.success, result.message
        assert result.cx_error_px is not None and result.cx_error_px <= 60.0, result.message
        assert result.cy_error_px is not None and result.cy_error_px <= 60.0, result.message

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extended_pinhole_reprojection_rms_is_low(self, seed):
        # 합성 데이터(노이즈 없음)이므로 RMS가 낮아야 한다 - 높게 나온다면
        # 코너 대응 관계가 잘못됐거나(순서 반전 등) 최적화가 발산한 것이다.
        result = run_extended_pinhole_accuracy_check(use_rational_model=False, seed=seed)
        assert result.success, result.message
        assert result.rms_error is not None and result.rms_error <= 1.5, result.message

    @pytest.mark.parametrize("seed", SEEDS)
    def test_pinhole_succeeds_with_zero_distortion(self, seed):
        # Pinhole은 왜곡을 추정하지 않으므로 fx/fy 정답 근접은 요구하지 않되
        # (모델 구조상 어느 정도 편향이 정상 - self_check.py 참고),
        # "왜곡 계수가 정말 전부 0으로 고정됐는가"와 "계산이 성공했는가"는
        # 반드시 확인한다.
        result = run_pinhole_accuracy_check(seed=seed)
        assert result.success, result.message
        assert result.passed, result.message


class TestRationalModelParameterCount:
    """(2) rational model 켰을 때 배열 길이/자유도가 맞는지 확인."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_extended_pinhole_has_5_free_params(self, seed):
        """기본값(꺼짐)은 k1,k2,p1,p2,k3 5계수만 추정해야 한다."""
        result = run_extended_pinhole_accuracy_check(use_rational_model=False, seed=seed)
        assert result.success, result.message
        assert result.distortion_array_length == 5, (
            f"기본 모드의 distortion 배열 길이는 5여야 하는데 {result.distortion_array_length}"
        )
        assert result.free_param_count == 5, (
            f"기본 모드에서 실제 추정된(0이 아닌) 계수는 5개여야 하는데 {result.free_param_count}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_rational_model_has_8_free_params(self, seed):
        """Rational model(체크박스 ON, --rational)은 k1~k6,p1,p2 총 8개
        자유도를 추정해야 한다.

        주의: OpenCV 버전에 따라 distortion 배열의 "길이" 자체는 8이 아니라
        14로 나올 수 있다(s1~s4, taux, tauy가 뒤에 항상 0으로 붙는 경우 -
        대화 중 OpenCV 4.13.0에서 실측 확인). 그래서 배열 길이를 8로
        하드코딩해서 assert하지 않고, "0이 아닌(실제로 추정된) 계수 개수"가
        8인지를 확인한다. 배열 길이는 5 또는 14 중 하나(현재 확인된 버전
        범위)여야 한다는 정도만 느슨하게 체크한다.
        """
        result = run_extended_pinhole_accuracy_check(use_rational_model=True, seed=seed)
        assert result.success, result.message
        assert result.free_param_count == 8, (
            f"rational model에서 실제 추정된(0이 아닌) 계수는 8개(k1~k6,p1,p2)여야 "
            f"하는데 {result.free_param_count}. distortion={result.distortion_array_length}칸"
        )
        assert result.distortion_array_length in (8, 14), (
            "rational model의 distortion 배열 길이가 예상 범위(8 또는 14)를 벗어났습니다 - "
            "OpenCV 버전이 바뀌면서 s1~s4/taux,tauy 처리 방식이 또 달라졌을 수 있습니다: "
            f"{result.distortion_array_length}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_rational_model_has_more_free_params_than_default(self, seed):
        """체크박스를 켜면 끈 것보다 자유도가 늘어나야 한다 - 이 프로젝트의
        핵심 요구사항(원래 대화의 발단)이 실제로 지켜지는지 직접 비교."""
        default_result = run_extended_pinhole_accuracy_check(use_rational_model=False, seed=seed)
        rational_result = run_extended_pinhole_accuracy_check(use_rational_model=True, seed=seed)
        assert default_result.success and rational_result.success
        assert rational_result.free_param_count > default_result.free_param_count, (
            f"Rational model을 켰는데 자유도가 늘지 않았습니다: "
            f"기본={default_result.free_param_count}, rational={rational_result.free_param_count}"
        )
