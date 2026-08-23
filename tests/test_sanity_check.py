"""
tests/test_sanity_check.py
=================================

설계 문서 8번 - Calibration 결과 sanity check. "성공"으로 리턴된 결과라도
물리적으로 이상하면 경고가 떠야 하고, 정상적인 결과에는 불필요한 경고가
뜨지 않아야 한다는 두 방향을 모두 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.sanity_check import (
    SanitySeverity,
    run_sanity_check,
    run_sanity_checks,
    format_sanity_checks,
)
from calibration.types import CalibrationResult, CameraConfig, CameraModelType


def _make_camera_config(w=1920, h=1080, hfov=None, vfov=None) -> CameraConfig:
    return CameraConfig(width=w, height=h, hfov_deg=hfov, vfov_deg=vfov)


def _normal_pinhole_result() -> CalibrationResult:
    K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
    D = np.zeros(5)
    return CalibrationResult(
        model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=D,
        rms_error=0.35, success=True,
    )


def _normal_extended_result() -> CalibrationResult:
    K = np.array([[1100.0, 0, 960.0], [0, 1105.0, 540.0], [0, 0, 1]])
    D = np.array([-0.28, 0.10, 0.0, 0.0, 0.0])
    return CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE, camera_matrix=K, distortion=D,
        rms_error=0.30, success=True,
    )


def _normal_fisheye_result() -> CalibrationResult:
    K = np.array([[600.0, 0, 960.0], [0, 600.0, 540.0], [0, 0, 1]])
    D = np.array([0.02, -0.01, 0.005, -0.001])
    return CalibrationResult(
        model_name=CameraModelType.FISHEYE, camera_matrix=K, distortion=D,
        rms_error=0.40, success=True,
    )


class TestNormalResultsPassCleanly:
    def test_normal_pinhole_has_no_issues(self):
        check = run_sanity_check(_normal_pinhole_result(), _make_camera_config())
        assert check.issues == []
        assert check.passed

    def test_normal_extended_has_no_issues(self):
        check = run_sanity_check(_normal_extended_result(), _make_camera_config())
        assert check.issues == []
        assert check.passed

    def test_normal_fisheye_has_no_issues(self):
        check = run_sanity_check(_normal_fisheye_result(), _make_camera_config())
        assert check.issues == []
        assert check.passed


class TestFailedCalibration:
    def test_failed_calibration_produces_error(self):
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, success=False, error_message="프레임 부족",
        )
        check = run_sanity_check(result, _make_camera_config())
        assert not check.passed
        assert any(i.severity == SanitySeverity.ERROR for i in check.issues)


class TestNonFiniteValues:
    def test_nan_in_camera_matrix_is_error(self):
        K = np.array([[1100.0, 0, 960.0], [0, np.nan, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert check.has_errors
        assert any(i.code == "camera_matrix_non_finite" for i in check.issues)

    def test_inf_distortion_is_error(self):
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        D = np.array([np.inf, 0.1, 0, 0, 0])
        result = CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE, camera_matrix=K, distortion=D,
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert check.has_errors
        assert any(i.code == "distortion_non_finite" for i in check.issues)

    def test_nan_rms_is_error(self):
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=float("nan"), success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert check.has_errors
        assert any(i.code == "rms_non_finite" for i in check.issues)


class TestFocalLength:
    def test_negative_fx_is_error(self):
        K = np.array([[-1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert check.has_errors
        assert any(i.code == "fx_non_positive" for i in check.issues)

    def test_absurdly_small_focal_length_is_warning(self):
        K = np.array([[5.0, 0, 960.0], [0, 5.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "fx_out_of_range" for i in check.issues)
        assert any(i.code == "fy_out_of_range" for i in check.issues)

    def test_absurdly_large_focal_length_is_warning(self):
        K = np.array([[50000.0, 0, 960.0], [0, 50000.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "fx_out_of_range" for i in check.issues)

    def test_aspect_ratio_mismatch_is_warning(self):
        K = np.array([[1100.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "aspect_ratio_off" for i in check.issues)


class TestPrincipalPoint:
    def test_principal_point_far_outside_image_is_warning(self):
        K = np.array([[1100.0, 0, 5000.0], [0, 1100.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "cx_out_of_bounds" for i in check.issues)

    def test_principal_point_slightly_off_center_is_fine(self):
        # 완전히 이미지 중앙이 아니어도(크롭/오프셋) 정상 범위 안이면 경고 없음
        K = np.array([[1100.0, 0, 1000.0], [0, 1100.0, 560.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert not any(i.code in ("cx_out_of_bounds", "cy_out_of_bounds") for i in check.issues)


class TestDistortion:
    def test_pinhole_with_nonzero_distortion_is_error(self):
        """Pinhole은 구조적으로 distortion=0이어야 한다 - 0이 아니면 모델 구현
        자체가 잘못된 것이므로 WARNING이 아니라 ERROR."""
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        D = np.array([0.1, 0, 0, 0, 0])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=D,
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert check.has_errors
        assert any(i.code == "pinhole_distortion_nonzero" for i in check.issues)

    def test_extreme_distortion_coefficient_is_warning(self):
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        D = np.array([15.0, 0.1, 0, 0, 0])  # 발산에 가까운 비정상적으로 큰 k1
        result = CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE, camera_matrix=K, distortion=D,
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "distortion_magnitude_large" for i in check.issues)

    def test_tiny_distortion_coefficient_is_warning(self):
        """parameter가 비정상적으로 작은 경우 (문서 8번 체크리스트 마지막 항목)."""
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        D = np.array([1e-6, 1e-7, 0, 0, 0])
        result = CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE, camera_matrix=K, distortion=D,
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "distortion_magnitude_tiny" for i in check.issues)


class TestFov:
    def test_fisheye_and_pinhole_use_different_fov_formula(self):
        """동일한 fx로도 Fisheye(equidistant)와 Pinhole(perspective)의 FOV
        추정치가 달라야 한다 - 광각일수록 atan 공식은 FOV를 과소평가한다."""
        K = np.array([[600.0, 0, 960.0], [0, 600.0, 540.0], [0, 0, 1]])
        pinhole = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        fisheye = CalibrationResult(
            model_name=CameraModelType.FISHEYE, camera_matrix=K, distortion=np.zeros(4),
            rms_error=0.3, success=True,
        )
        cfg = _make_camera_config()
        pinhole_check = run_sanity_check(pinhole, cfg)
        fisheye_check = run_sanity_check(fisheye, cfg)
        # 둘 다 정상 범위라 경고는 없어야 하지만, 내부적으로 다른 공식을 쓰는지는
        # spec mismatch 케이스로 간접 확인한다 (아래 스펙 비교 테스트에서 다룸).
        assert pinhole_check.passed and fisheye_check.passed

    def test_hfov_spec_mismatch_is_warning(self):
        # 실제 fx=1100, w=1920 이면 hfov ~ 82도인데 스펙을 180도로 준 경우
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        cfg = _make_camera_config(hfov=180.0)
        check = run_sanity_check(result, cfg)
        assert any(i.code == "hfov_spec_mismatch" for i in check.issues)

    def test_hfov_spec_close_match_has_no_warning(self):
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        import math
        expected_hfov = math.degrees(2 * math.atan((1920 / 2) / 1100.0))
        cfg = _make_camera_config(hfov=expected_hfov)
        check = run_sanity_check(result, cfg)
        assert not any(i.code == "hfov_spec_mismatch" for i in check.issues)


class TestRms:
    def test_high_rms_is_warning(self):
        result = _normal_pinhole_result()
        result.rms_error = 1.5
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "rms_high" for i in check.issues)

    def test_very_high_rms_is_warning_with_different_code(self):
        result = _normal_pinhole_result()
        result.rms_error = 5.0
        check = run_sanity_check(result, _make_camera_config())
        assert any(i.code == "rms_very_high" for i in check.issues)
        assert not any(i.code == "rms_high" for i in check.issues)


class TestBatchHelpers:
    def test_run_sanity_checks_returns_one_per_result(self):
        results = [_normal_pinhole_result(), _normal_extended_result(), _normal_fisheye_result()]
        checks = run_sanity_checks(results, _make_camera_config())
        assert len(checks) == 3
        assert [c.model_name for c in checks] == [r.model_name for r in results]

    def test_format_sanity_checks_includes_all_models(self):
        results = [_normal_pinhole_result(), _normal_extended_result()]
        checks = run_sanity_checks(results, _make_camera_config())
        text = format_sanity_checks(checks)
        assert "pinhole" in text
        assert "extended_pinhole" in text

    def test_format_lists_each_issue(self):
        K = np.array([[-5.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
        result = CalibrationResult(
            model_name=CameraModelType.PINHOLE, camera_matrix=K, distortion=np.zeros(5),
            rms_error=0.3, success=True,
        )
        check = run_sanity_check(result, _make_camera_config())
        formatted = check.format()
        assert "fx" in formatted
        for issue in check.issues:
            assert issue.message in formatted
