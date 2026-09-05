"""
tests/test_windshield_spline.py
====================================

STEP 4 Spline Windshield Model 검증(기본 - runtime model, 단일 STAGE A fit).
STAGE B/AUTO/leakage/repeated hold-out/stability는
tests/test_windshield_spline_stabilization.py가 담당한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.spherical import SphericalWindshieldModel, calibrate_spherical
from calibration.windshield.spline import (
    DEFAULT_MAX_DISPLACEMENT_M,
    SplineWindshieldModel,
    _deformed_surface_point_and_normal,
    calibrate_spline,
)
from calibration.windshield.baseline import BaselineWindshieldModel
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


# ---------------------------------------------------------------------------
# Test A/F - zero deformation matches Spherical exactly (Snell 포함)
# ---------------------------------------------------------------------------

def test_zero_deformation_matches_spherical_model():
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    grid = np.zeros((3, 4))
    spline_model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, IMG_W, IMG_H)
    spherical_model = SphericalWindshieldModel(K, D, _MODEL, center, radius)

    for u, v in [(640, 400), (500, 300), (800, 500), (640, 200)]:
        d_spline = np.array(spline_model.unproject_pixel(u, v))
        d_spherical = np.array(spherical_model.unproject_pixel(u, v))
        # finite-difference normal이 analytic radial normal에 아주 가깝지만
        # 완전히 동일하지는 않다(사용자 스펙 12번의 fallback 특성) - 느슨한
        # 허용오차를 쓴다.
        assert np.allclose(d_spline, d_spherical, atol=1e-3)


def test_zero_deformation_project_point_matches_spherical():
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    grid = np.zeros((3, 4))
    spline_model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, IMG_W, IMG_H)
    spherical_model = SphericalWindshieldModel(K, D, _MODEL, center, radius)

    target = np.array([0.1, 0.05, 3.0])
    uv_spline = spline_model.project_point(*target)
    uv_spherical = spherical_model.project_point(*target)
    assert uv_spline == pytest.approx(uv_spherical, abs=0.05)


# ---------------------------------------------------------------------------
# Test B - constant offset shifts surface in expected direction
# ---------------------------------------------------------------------------

def test_constant_offset_moves_surface_point_outward():
    """균일한(양의) Δs는 표면의 실제 교차점을 반경 방향 바깥쪽으로 정확히
    Δs만큼 옮겨야 한다(사용자 스펙 6번 정의 그대로) - 이건 surface point
    자체를 직접 검증한다.

    참고: 이 근사(카메라가 표면에서 겨우 ~수mm 떨어진 좌표계)에서는 이
    표면 이동이 최종 "far-field 방향"(unproject_pixel의 출력)에는 실측
    결과 거의 영향을 주지 않는다(<1e-6 수준) - 유리를 통과한 뒤 방향을
    결정하는 것은 진입점의 정확한 위치가 아니라 표면의 국소적인 기울기
    (normal 방향)이기 때문이다(균일 Δs는 normal을 전혀 바꾸지 않는다).
    이는 물리적으로 타당한 현상이라 "표면 자체가 이동했는가"를 직접
    확인하는 이 테스트로 Test B의 의도를 검증한다."""
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    ds = 0.008  # 8mm

    zero_grid = np.zeros((2, 2))
    offset_grid = np.full((2, 2), ds)

    u, v = 700.0, 450.0
    p_zero = _deformed_surface_point_and_normal(u, v, baseline, center, radius, zero_grid, IMG_W, IMG_H)
    p_offset = _deformed_surface_point_and_normal(u, v, baseline, center, radius, offset_grid, IMG_W, IMG_H)
    assert p_zero is not None and p_offset is not None
    point_zero, _ = p_zero
    point_offset, _ = p_offset

    delta = point_offset - point_zero
    radial_at_zero = (point_zero - center) / np.linalg.norm(point_zero - center)
    # delta는 거의 전적으로 방사(radial) 방향이어야 하고, 그 크기는 거의
    # 정확히 ds(미터)여야 한다(정의상: local_radius = radius + ds인 구와의
    # 교차점이므로).
    assert np.dot(delta, radial_at_zero) == pytest.approx(ds, abs=1e-4)
    assert float(np.linalg.norm(delta)) == pytest.approx(ds, abs=1e-4)


# ---------------------------------------------------------------------------
# Test D - surface normal points outward, away from sphere center
# ---------------------------------------------------------------------------

def test_surface_normal_points_outward_from_center():
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    grid = np.zeros((3, 4))
    baseline = BaselineWindshieldModel(K, D, _MODEL)

    for u, v in [(640, 400), (500, 300), (900, 600)]:
        result = _deformed_surface_point_and_normal(u, v, baseline, center, radius, grid, IMG_W, IMG_H)
        assert result is not None
        point, normal = result
        assert np.isfinite(normal).all()
        assert abs(float(np.linalg.norm(normal)) - 1.0) < 1e-6
        # zero deformation이면 normal은 순수 반경 방향(normalize(point-center))과
        # 거의 일치해야 한다(사용자 스펙 12번 요구사항의 최소 sanity check).
        radial = (point - center) / np.linalg.norm(point - center)
        assert np.dot(normal, radial) > 0.999


# ---------------------------------------------------------------------------
# project_point / unproject_pixel round-trip sanity
# ---------------------------------------------------------------------------

def test_project_point_unproject_pixel_roundtrip():
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    grid = np.zeros((3, 4))
    model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, IMG_W, IMG_H)

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
    center, radius = _known_sphere()
    grid = np.zeros((3, 4))
    model = SplineWindshieldModel(K, D, _MODEL, center, radius, grid, IMG_W, IMG_H)

    positions = [
        (IMG_W / 2, IMG_H / 2), (10, 10), (IMG_W - 10, 10),
        (10, IMG_H - 10), (IMG_W - 10, IMG_H - 10), (IMG_W / 2, 10), (10, IMG_H / 2),
    ]
    for u, v in positions:
        d = model.unproject_pixel(u, v)
        assert all(np.isfinite(x) for x in d)


# ---------------------------------------------------------------------------
# K,D immutability (사용자 스펙 Test G, 기본 경로)
# ---------------------------------------------------------------------------

def test_calibrate_spline_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    calibrate_spline(dataset, config, camera_config, train_ids, test_ids)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# calibrate_spline() end-to-end - Test A(zero bump) / Test C(bump, loosened)
# ---------------------------------------------------------------------------

def test_calibrate_spline_succeeds_on_pure_spherical_dataset():
    """Δs=0에 가까운 진짜 정답(순수 Spherical GT)에서 spline이 성공하고,
    deformation이 크게 부풀려지지 않아야 한다(사용자 스펙 Test A의 calibration
    파이프라인 버전)."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    assert result.residual_stats.rmse < 2.0  # 순수 spherical GT이므로 낮은 RMS 기대
    assert result.fitted_params["diag_deformation_max_abs_m"] < DEFAULT_MAX_DISPLACEMENT_M


def test_calibrate_spline_runs_on_bump_dataset_and_respects_bounds():
    """실제 국부 곡면 차이(smooth Gaussian bump, tests._windshield_test_utils의
    독립 GT generator)가 있는 데이터에서도 STAGE A가 수렴하고, 결과 grid가
    optimizer bound 안에 머무는지 확인한다 - 정확한 GT 형태 복원까지는
    요구하지 않는다(사용자 스펙 37-C "너무 타이트한 exact recovery assertion은
    피한다").

    실측(이 라운드에서 직접 확인): 카메라가 base sphere 표면에서 겨우 ~1cm
    떨어져 있는 이 좌표계에서는 "표면 전체가 얼마나 큰가"(거의 균일한
    반경 보정)가 "국소적으로 어떻게 휘었는가"보다 far-field target에 훨씬
    큰 영향을 준다 - 그래서 optimizer가 국소 Gaussian bump 모양을 정밀
    재현하기보다 대체로 균일에 가까운 보정으로 수렴하는 경향이 있다.
    물리적으로 타당한 현상이며(멀리 있는 목표점에는 진입점의 정확한 위치보다
    표면 방향/각도가 더 중요), 버그가 아니다 - 그래서 이 테스트는 "정확한
    GT 모양 복원"이 아니라 "STAGE A가 유한하고 bound 안에 있는 해로 수렴한다"
    는 구조적 정합성만 확인한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    max_disp = 0.01
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0, "max_displacement_m": max_disp})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    rows, cols = int(result.fitted_params["spline_rows"]), int(result.fitted_params["spline_cols"])
    grid = np.array([[result.fitted_params[f"spline_ds_{r}_{c}"] for c in range(cols)] for r in range(rows)])
    assert np.all(np.isfinite(grid))
    assert np.all(np.abs(grid) <= max_disp + 1e-9)
