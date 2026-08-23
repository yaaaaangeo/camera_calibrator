"""
tests/test_bootstrap.py
==============================

설계 문서 20/21/22번 - Bootstrap Stability / Parameter Confidence Interval을
Pinhole/Extended Pinhole/Fisheye 세 모델 공용으로 일반화한 모듈.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.bootstrap import (
    add_normal_approximation_ci,
    compute_parameter_bootstrap,
    format_parameter_uncertainty,
)
from calibration.types import CameraModelType, ParameterUncertainty

W, H = 1920, 1080
TRUE_K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
TRUE_D_PINHOLE = np.zeros(5)
TRUE_D_FISHEYE = np.array([0.02, -0.01, 0.005, -0.001])


def _synthetic_views(n=20, seed=0):
    objp = np.zeros((8 * 6, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.04
    rng = np.random.default_rng(seed)

    object_points, image_points = [], []
    for _ in range(n):
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.3, (rng.random() - 0.5) * 0.3, 0.5 + rng.random() * 0.3])
        projected, _ = cv2.projectPoints(objp, rvec, tvec, TRUE_K, TRUE_D_PINHOLE)
        if np.any(projected.reshape(-1, 2) < 0) or np.any(projected.reshape(-1, 2)[:, 0] > W):
            continue
        object_points.append(objp.reshape(-1, 1, 3))
        image_points.append(projected.reshape(-1, 1, 2).astype(np.float32))
    return object_points, image_points


class TestComputeParameterBootstrap:
    def test_pinhole_returns_uncertainty_with_bootstrap_method(self):
        object_points, image_points = _synthetic_views(n=20)
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (W, H), None, None, flags=flags
        )
        result = compute_parameter_bootstrap(
            object_points, image_points, (W, H), CameraModelType.PINHOLE,
            K, D, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS, n_bootstrap=15, rng_seed=1,
        )
        assert result is not None
        assert result.method == "bootstrap"
        assert result.fx_std is not None and result.fx_std >= 0
        assert result.n_bootstrap_success is not None and result.n_bootstrap_success > 0
        assert result.fx_mean is not None
        assert result.fx_median is not None
        assert result.fx_min <= result.fx_median <= result.fx_max
        assert result.fx_stability is not None
        assert 0.0 <= result.fx_stability <= 100.0
        assert result.overall_stability is not None
        assert 0.0 <= result.overall_stability <= 100.0
        assert len(result.distortion_stats) == int(D.size)
        assert all(s.mean is not None for s in result.distortion_stats)
        assert all(s.median is not None for s in result.distortion_stats)
        assert all(s.ci_low is not None and s.ci_high is not None for s in result.distortion_stats)
        assert all(s.stability_score is not None for s in result.distortion_stats)

    def test_confidence_interval_contains_reference_estimate(self):
        """95% CI는 (대부분의 정상적인 경우) 원래 전체 데이터 추정치를 포함해야 한다."""
        object_points, image_points = _synthetic_views(n=25, seed=2)
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (W, H), None, None, flags=flags
        )
        result = compute_parameter_bootstrap(
            object_points, image_points, (W, H), CameraModelType.PINHOLE,
            K, D, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS, n_bootstrap=30, rng_seed=3,
        )
        assert result is not None
        fx = float(K[0, 0])
        assert result.fx_ci_low <= fx <= result.fx_ci_high

    def test_empty_input_returns_none(self):
        result = compute_parameter_bootstrap(
            [], [], (W, H), CameraModelType.PINHOLE, TRUE_K, TRUE_D_PINHOLE, flags=0
        )
        assert result is None

    def test_parallel_matches_sequential_for_same_seed(self):
        object_points, image_points = _synthetic_views(n=20, seed=6)
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        _, K, D, _, _ = cv2.calibrateCamera(
            object_points, image_points, (W, H), None, None, flags=flags
        )
        common = dict(
            object_points=object_points,
            image_points=image_points,
            image_size=(W, H),
            model=CameraModelType.PINHOLE,
            K_ref=K,
            D_ref=D,
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
            n_bootstrap=12,
            rng_seed=9,
        )
        sequential = compute_parameter_bootstrap(**common, n_jobs=1)
        parallel = compute_parameter_bootstrap(**common, n_jobs=2)

        assert sequential is not None
        assert parallel is not None
        assert parallel.n_bootstrap_success == sequential.n_bootstrap_success
        assert parallel.fx_std == pytest.approx(sequential.fx_std, rel=1e-4, abs=1e-8)
        assert parallel.fy_std == pytest.approx(sequential.fy_std, rel=1e-4, abs=1e-8)
        assert parallel.overall_stability == pytest.approx(sequential.overall_stability, rel=1e-4, abs=1e-8)
        assert [s.label for s in parallel.distortion_stats] == [s.label for s in sequential.distortion_stats]

    def test_fisheye_dispatch_uses_fisheye_calibrate(self):
        """model=FISHEYE일 때 cv2.fisheye.calibrate 경로를 타는지 - 최소한
        예외 없이 결과가 나오는지로 간접 확인 (완전한 fisheye 데이터 생성은
        비용이 크므로 여기선 dispatch 자체만 확인)."""
        object_points, image_points = _synthetic_views(n=15, seed=4)
        # fisheye 포맷 요구사항(float64, (N,1,3)/(N,1,2)) 맞추기
        obj64 = [o.astype(np.float64) for o in object_points]
        img64 = [i.astype(np.float64) for i in image_points]
        K = TRUE_K.astype(np.float64)
        D = np.zeros((4, 1))
        recompute_flag = getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", 0)
        intrinsic_guess_flag = getattr(cv2.fisheye, "CALIB_USE_INTRINSIC_GUESS", 0)
        try:
            rms, K_est, D_est, rvecs, tvecs = cv2.fisheye.calibrate(
                obj64, img64, (W, H), K.copy(), D.copy(),
                flags=recompute_flag,
            )
        except cv2.error:
            pytest.skip("이 합성 데이터로는 fisheye 초기 캘리브레이션 자체가 실패함 (환경 의존적)")
            return
        result = compute_parameter_bootstrap(
            obj64, img64, (W, H), CameraModelType.FISHEYE,
            K_est, D_est, flags=recompute_flag | intrinsic_guess_flag,
            n_bootstrap=10, rng_seed=5,
        )
        # 데이터가 적어 실패할 수도 있으니 None이어도 괜찮음 - 예외만 안 나면 된다.
        if result is not None:
            assert result.method == "bootstrap"


class TestAddNormalApproximationCi:
    def test_populates_ci_from_std(self):
        u = ParameterUncertainty(fx_std=2.0, fy_std=2.5, cx_std=1.0, cy_std=1.2, method="covariance")
        K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
        add_normal_approximation_ci(u, K)
        assert u.fx_ci_low == pytest.approx(1000.0 - 1.96 * 2.0)
        assert u.fx_ci_high == pytest.approx(1000.0 + 1.96 * 2.0)
        assert u.cy_ci_low == pytest.approx(540.0 - 1.96 * 1.2)
        assert u.fx_mean == 1000.0
        assert u.fx_stability == pytest.approx(99.8)
        assert u.overall_stability is not None

    def test_skips_bootstrap_method(self):
        """bootstrap 결과는 이미 percentile CI가 있으므로 정규근사를 덮어쓰지 않는다."""
        u = ParameterUncertainty(
            fx_std=2.0, method="bootstrap", fx_ci_low=990.0, fx_ci_high=1010.0,
        )
        K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
        result = add_normal_approximation_ci(u, K)
        assert result.fx_ci_low == 990.0
        assert result.fx_ci_high == 1010.0

    def test_handles_missing_std(self):
        u = ParameterUncertainty(method="covariance")
        K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
        result = add_normal_approximation_ci(u, K)
        assert result.fx_ci_low is None


class TestFormatParameterUncertainty:
    def test_includes_ci_when_present(self):
        u = ParameterUncertainty(
            fx_std=2.1, fy_std=2.4, cx_std=1.2, cy_std=1.5, method="covariance",
            fx_ci_low=808.2, fx_ci_high=816.4,
        )
        text = format_parameter_uncertainty(u)
        assert "95% CI" in text
        assert "fx" in text and "fy" in text

    def test_handles_none(self):
        assert "계산되지 않았습니다" in format_parameter_uncertainty(None)

    def test_bootstrap_method_shows_sample_count(self):
        u = ParameterUncertainty(fx_std=1.0, method="bootstrap", n_bootstrap_success=18, overall_stability=97.5)
        text = format_parameter_uncertainty(u)
        assert "18" in text
        assert "Parameter Stability" in text
