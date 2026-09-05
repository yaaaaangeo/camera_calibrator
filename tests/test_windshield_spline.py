"""
tests/test_windshield_spline.py
====================================

STEP 4 Spline Windshield Model 검증(기본 - runtime model, 단일 STAGE A fit).
STAGE B/AUTO/leakage/repeated hold-out/stability는
tests/test_windshield_spline_stabilization.py가 담당한다.

이 모델은 코너마다 2번(inner+outer)의 3-unknown ray-surface intersection
solve가 필요해 Grid/RBF보다 근본적으로 계산이 무겁다 - `calibrate_spline()`
1회 호출이 최소 grid(4x4)에서도 수 분 걸릴 수 있다(analytic Jacobian +
재사용 가능한 B-spline basis evaluator로 이미 크게 최적화했지만, 물리적으로
정확한 surface intersection 자체의 비용은 줄일 수 없다). 그래서 이 파일은
비용이 큰 `calibrate_spline()` 호출을 모듈 스코프 fixture로 묶어 여러
assertion이 같은 결과를 재사용하게 한다 - 각 테스트 함수가 독립적으로
다시 계산하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.spherical import SphericalWindshieldModel
from calibration.windshield.spline import (
    DEFAULT_MAX_DISPLACEMENT_M,
    MIN_SPLINE_GRID_SIZE,
    SplineWindshieldModel,
    calibrate_spline,
    compute_angular_fov_scale,
    evaluate_inner_surface,
    _build_spline_basis,
)
from tests._windshield_test_utils import (
    IMG_H,
    IMG_W,
    build_synthetic_spherical_windshield_dataset,
    build_synthetic_spline_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _config(K, D, **kwargs) -> WindshieldConfig:
    return WindshieldConfig(
        base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPLINE, **kwargs,
    )


def _known_sphere():
    # 안전한 margin(is_valid_spherical_windshield의 MIN_SPHERE_MARGIN_M=0.01,
    # MIN_STANDOFF_M=0.01보다 충분히 여유 있게)을 두고 고른 값 - 경계에 딱
    # 붙은 값을 쓰면 정상적인 단위 테스트조차 물리적 유효성 검사에 걸릴 수 있다.
    center = np.array([0.0, 0.0, -4.9])
    radius = 5.0
    return center, radius


def _known_scales(K, D):
    center, radius = _known_sphere()
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    theta_scale, phi_scale = compute_angular_fov_scale(baseline, center, radius, IMG_W, IMG_H)
    return center, radius, theta_scale, phi_scale, baseline


# ---------------------------------------------------------------------------
# Test A/F - zero deformation matches Spherical exactly (Snell 포함)
# ---------------------------------------------------------------------------

def test_zero_deformation_matches_spherical_model():
    K, D = default_camera_matrix_distortion()
    center, radius, theta_scale, phi_scale, _baseline = _known_scales(K, D)
    grid = np.zeros((MIN_SPLINE_GRID_SIZE, MIN_SPLINE_GRID_SIZE))
    spline_model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, theta_scale, phi_scale)
    spherical_model = SphericalWindshieldModel(K, D, _MODEL, center, radius)

    for u, v in [(640, 400), (500, 300), (800, 500), (640, 200)]:
        d_spline = np.array(spline_model.unproject_pixel(u, v))
        d_spherical = np.array(spherical_model.unproject_pixel(u, v))
        # Delta_s=0이면 inner surface가 정확히 base sphere이고 analytic
        # normal도 정확한 radial 방향과 수학적으로 일치해야 한다 - finite
        # difference를 쓰던 이전 구현(허용오차 1e-3)과 달리 이제는 거의
        # 수치오차 수준(1e-8)까지 일치한다.
        assert np.allclose(d_spline, d_spherical, atol=1e-6)


def test_zero_deformation_project_point_matches_spherical():
    K, D = default_camera_matrix_distortion()
    center, radius, theta_scale, phi_scale, _baseline = _known_scales(K, D)
    grid = np.zeros((MIN_SPLINE_GRID_SIZE, MIN_SPLINE_GRID_SIZE))
    spline_model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, theta_scale, phi_scale)
    spherical_model = SphericalWindshieldModel(K, D, _MODEL, center, radius)

    target = np.array([0.1, 0.05, 3.0])
    uv_spline = spline_model.project_point(*target)
    uv_spherical = spherical_model.project_point(*target)
    assert uv_spline == pytest.approx(uv_spherical, abs=0.01)


# ---------------------------------------------------------------------------
# Test B - constant offset shifts surface in expected direction
# ---------------------------------------------------------------------------

def test_constant_offset_moves_surface_point_outward():
    """균일한(양의) Delta_s는 표면의 실제 점을 정확히 S0 + ds*N0 방향으로
    옮겨야 한다(사용자 스펙 9번 정의 그대로) - `evaluate_inner_surface`를
    직접 호출해 검증한다(픽셀->ray 변환을 거치지 않는 가장 직접적인 단위
    테스트)."""
    K, D = default_camera_matrix_distortion()
    center, radius, theta_scale, phi_scale, _baseline = _known_scales(K, D)
    ds = 0.008  # 8mm
    n = MIN_SPLINE_GRID_SIZE
    zero_grid = np.zeros((n, n))
    offset_grid = np.full((n, n), ds)
    basis = _build_spline_basis(n, n, 3)

    p, q = 0.1, -0.2
    ev_zero = evaluate_inner_surface(p, q, center, radius, zero_grid, theta_scale, phi_scale, basis)
    ev_offset = evaluate_inner_surface(p, q, center, radius, offset_grid, theta_scale, phi_scale, basis)
    assert ev_zero is not None and ev_offset is not None

    delta = ev_offset.point - ev_zero.point
    radial_at_zero = (ev_zero.point - center) / np.linalg.norm(ev_zero.point - center)
    # 균일 grid에서는 Delta_s(p,q)=ds(상수)이므로 delta는 정확히 ds*n0
    # 방향/크기여야 한다.
    assert np.dot(delta, radial_at_zero) == pytest.approx(ds, abs=1e-9)
    assert float(np.linalg.norm(delta)) == pytest.approx(ds, abs=1e-9)
    # 균일 offset은 normal 자체를 바꾸지 않는다(접선 미분이 동일) - 물리적으로
    # 타당한 현상이며 이 테스트로 명시적으로 확인한다.
    assert np.allclose(ev_zero.normal, ev_offset.normal, atol=1e-9)


# ---------------------------------------------------------------------------
# Test D - surface normal points outward, away from sphere center
# ---------------------------------------------------------------------------

def test_surface_normal_points_outward_from_center():
    K, D = default_camera_matrix_distortion()
    center, radius, theta_scale, phi_scale, _baseline = _known_scales(K, D)
    n = MIN_SPLINE_GRID_SIZE
    grid = np.zeros((n, n))
    basis = _build_spline_basis(n, n, 3)

    for p, q in [(0.0, 0.0), (0.3, -0.5), (-0.6, 0.4)]:
        ev = evaluate_inner_surface(p, q, center, radius, grid, theta_scale, phi_scale, basis)
        assert ev is not None
        assert np.isfinite(ev.normal).all()
        assert abs(float(np.linalg.norm(ev.normal)) - 1.0) < 1e-9
        # Delta_s=0이면 normal은 정확히 순수 반경 방향(normalize(point-center))
        # 이어야 한다(analytic 미분 - finite difference 근사가 아니다).
        radial = (ev.point - center) / np.linalg.norm(ev.point - center)
        assert np.dot(ev.normal, radial) > 1.0 - 1e-9


# ---------------------------------------------------------------------------
# project_point / unproject_pixel round-trip sanity
# ---------------------------------------------------------------------------

def test_project_point_unproject_pixel_roundtrip():
    K, D = default_camera_matrix_distortion()
    center, radius, theta_scale, phi_scale, _baseline = _known_scales(K, D)
    n = MIN_SPLINE_GRID_SIZE
    grid = np.zeros((n, n))
    model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, theta_scale, phi_scale)

    u0, v0 = 640.0, 400.0
    d = np.array(model.unproject_pixel(u0, v0))
    target = d * 3.0
    u1, v1 = model.project_point(*target)
    assert (u1, v1) == pytest.approx((u0, v0), abs=0.5)


# ---------------------------------------------------------------------------
# Boundary/NaN-free checks (사용자 스펙 Test D 확장)
# ---------------------------------------------------------------------------

def test_no_nan_inf_at_center_and_corners_and_edges():
    K, D = default_camera_matrix_distortion()
    center, radius, theta_scale, phi_scale, _baseline = _known_scales(K, D)
    n = MIN_SPLINE_GRID_SIZE
    grid = np.zeros((n, n))
    model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, theta_scale, phi_scale)

    positions = [
        (IMG_W / 2, IMG_H / 2), (10, 10), (IMG_W - 10, 10),
        (10, IMG_H - 10), (IMG_W - 10, IMG_H - 10), (IMG_W / 2, 10), (10, IMG_H / 2),
    ]
    for u, v in positions:
        d = model.unproject_pixel(u, v)
        assert all(np.isfinite(x) for x in d)


# ---------------------------------------------------------------------------
# calibrate_spline() end-to-end - 비용이 큰 호출은 모듈 fixture로 1회만 실행
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spherical_gt_spline_result():
    """순수 Spherical GT(Delta_s=0이 정답)에서 calibrate_spline()을 1번만
    실행하고 여러 테스트가 그 결과를 공유한다 - Test A(zero bump 경로)와
    K,D immutability(Test G)를 이 fixture 하나로 함께 검증한다."""
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)
    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)
    return result, K, D, K_before, D_before


def test_calibrate_spline_succeeds_on_pure_spherical_dataset(spherical_gt_spline_result):
    """Delta_s~0에 가까운 진짜 정답(순수 Spherical GT)에서 spline이 성공하고,
    deformation이 크게 부풀려지지 않아야 한다(사용자 스펙 Test A의 calibration
    파이프라인 버전)."""
    result, *_ = spherical_gt_spline_result
    assert result.success, result.error_message
    assert result.residual_stats.rmse < 2.0  # 순수 spherical GT이므로 낮은 RMS 기대
    assert result.fitted_params["diag_deformation_max_abs_m"] < DEFAULT_MAX_DISPLACEMENT_M
    # 새 physics 표식(사용자 스펙 40번) - 실제로 Bicubic B-Spline normal-offset
    # 경로를 탔는지 fitted_params에서 확인.
    assert result.fitted_params["spline_surface_representation_is_bicubic_bspline"] == 1.0
    assert result.fitted_params["spline_degree"] == 3.0
    assert "spline_theta_scale_rad" in result.fitted_params
    assert "spline_phi_scale_rad" in result.fitted_params


def test_calibrate_spline_keeps_base_intrinsics_immutable(spherical_gt_spline_result):
    _, K, D, K_before, D_before = spherical_gt_spline_result
    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_calibrate_spline_runs_on_bump_dataset_and_respects_bounds():
    """실제 국부 곡면 차이(smooth Gaussian bump, tests._windshield_test_utils의
    독립 GT generator - 3D 방향 각도 거리 기반, production의 (p,q) pixel-FOV
    convention과 무관)가 있는 데이터에서도 STAGE A가 수렴하고, 결과 grid가
    optimizer bound 안에 머무는지 확인한다 - 정확한 GT 형태 복원까지는
    요구하지 않는다(사용자 스펙 34번 "너무 타이트한 exact recovery
    assertion은 피한다")."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    max_disp = 0.01
    config = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0, "max_displacement_m": max_disp})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    rows, cols = int(result.fitted_params["spline_rows"]), int(result.fitted_params["spline_cols"])
    grid = np.array([[result.fitted_params[f"spline_ds_{r}_{c}"] for c in range(cols)] for r in range(rows)])
    assert np.all(np.isfinite(grid))
    assert np.all(np.abs(grid) <= max_disp + 1e-9)
    # 완전히 아무 것도 하지 않은 건 아니라는 최소 sanity check(사용자 스펙
    # 34번 "Recovered deformation nonzero").
    assert np.max(np.abs(grid)) > 1e-6
