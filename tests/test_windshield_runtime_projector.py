"""
tests/test_windshield_runtime_projector.py
===============================================

Phase C-1 안정화 - Batch/LUT Windshield Runtime Projector 검증.

Exact point-by-point WindshieldModel(`build_projector()`)은 그대로 두고,
`calibration/windshield/runtime_projector.py`가 추가한 빠른 LUT 근사
계층(`build_runtime_projector()`)이 (1) exact 결과를 절대 바꾸지 않고,
(2) 스스로 주장한 정확도 tradeoff(가까운 포인트일수록 parallax 근사 오차가
커짐)를 실제로 보이는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.runtime_projector import (
    RuntimeWindshieldProjector,
    build_runtime_projector,
    project_points_exact_batch,
    unproject_pixels_exact_batch,
    validate_runtime_projector_vs_exact,
)
from calibration.windshield.spherical import SphericalWindshieldModel
from tests._windshield_test_utils import (
    DEFAULT_SPHERE_CENTER,
    DEFAULT_SPHERE_RADIUS,
    IMG_H,
    IMG_W,
    default_camera_matrix_distortion,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _sample_pixels(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    us = rng.uniform(0, IMG_W, n)
    vs = rng.uniform(0, IMG_H, n)
    return np.stack([us, vs], axis=1)


def _sample_points_at_distance(distance_m: float, n: int, seed: int = 0) -> np.ndarray:
    """카메라 정면 방향 기준 좁은 원뿔(FOV) 안에서 방향을 뽑아 지정된
    거리로 스케일한 3D 포인트를 만든다 - LiDAR 포인트 분포를 흉내낸다."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-0.3, 0.3, n)
    ys = rng.uniform(-0.2, 0.2, n)
    zs = np.ones(n)
    dirs = np.stack([xs, ys, zs], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs * distance_m


def test_runtime_projector_unproject_matches_exact_closely_for_baseline():
    """Baseline은 근사가 전혀 없는(unproject_pixel이 순수 (u,v)의 함수)
    경우이므로, LUT bilinear 보간 오차만 남고 root-solve 관련 오차는 없어야
    한다 - 각도 오차가 아주 작아야 한다."""
    K, D = default_camera_matrix_distortion()
    model = BaselineWindshieldModel(K, D, _MODEL)
    runtime = RuntimeWindshieldProjector(model, IMG_W, IMG_H)

    report = validate_runtime_projector_vs_exact(
        runtime,
        sample_points_xyz=_sample_points_at_distance(5.0, 20),
        sample_pixels_uv=_sample_pixels(300),
    )
    assert report.unproject_p99_deg < 1.0


def test_runtime_projector_project_points_matches_exact_for_undistorted_pinhole_at_any_distance():
    """왜곡 없는 순수 pinhole에서는 project_point가 방향에만 의존한다
    (scale-invariant projection - parallax 자체가 없음). 따라서 LUT의
    "방향만 쓰는" 근사가 거리와 무관하게 실제로 정확해야 한다."""
    K, D = default_camera_matrix_distortion()
    model = BaselineWindshieldModel(K, np.zeros_like(D), CameraModelType.PINHOLE)
    runtime = RuntimeWindshieldProjector(model, IMG_W, IMG_H, lut_rows=128, lut_cols=128)

    for distance in (1.0, 5.0, 50.0):
        report = validate_runtime_projector_vs_exact(
            runtime,
            sample_points_xyz=_sample_points_at_distance(distance, 50, seed=int(distance)),
            sample_pixels_uv=_sample_pixels(5),
        )
        assert report.project_p99_px < 5.0, f"distance={distance}m에서 근사 오차가 예상보다 큼"


def test_runtime_projector_shows_larger_error_for_closer_points_under_spherical_refraction():
    """Spherical(실제 굴절 - windshield 표면에서 parallax 발생)에서는,
    windshield에 훨씬 가까운 포인트일수록 "방향만" 쓰는 LUT 근사의 오차가
    먼 포인트(LiDAR 전형적 거리)보다 커야 한다 - 모듈이 스스로 문서화한
    tradeoff를 실측으로 확인한다."""
    # LUT 해상도를 충분히 높여서(200x200) bilinear 보간 자체의 잡음이
    # parallax 신호를 가리지 않게 한다 - 낮은 해상도(예: 96x96)에서는 두
    # 거리의 오차 차이가 보간 잡음에 묻혀 버림을 직접 확인했다(개발 중
    # 재현). near 거리는 windshield 안쪽 표면(z~0.3m)에 최대한 가깝게,
    # far 거리는 LiDAR 전형적 원거리로 크게 벌린다.
    K, D = default_camera_matrix_distortion()
    model = SphericalWindshieldModel(K, D, _MODEL, DEFAULT_SPHERE_CENTER, DEFAULT_SPHERE_RADIUS)
    runtime = RuntimeWindshieldProjector(model, IMG_W, IMG_H, lut_rows=200, lut_cols=200)

    near_report = validate_runtime_projector_vs_exact(
        runtime,
        sample_points_xyz=_sample_points_at_distance(0.35, 60, seed=1),
        sample_pixels_uv=_sample_pixels(5),
    )
    far_report = validate_runtime_projector_vs_exact(
        runtime,
        sample_points_xyz=_sample_points_at_distance(200.0, 60, seed=1),
        sample_pixels_uv=_sample_pixels(5),
    )
    # 단순 <=가 아니라 여유 있는 margin으로 비교해서, 근사 오차의 잔여
    # 보간 잡음(거리와 무관) 때문에 생기는 우연한 뒤집힘을 배제한다.
    assert far_report.project_median_px < near_report.project_median_px * 0.9


def test_runtime_projector_never_mutates_exact_model_behavior():
    """LUT를 만들거나 조회해도 exact_model 자체의 point-by-point 결과는
    절대 바뀌면 안 된다(사용자 스펙 - exact projector를 절대 바꾸지 않는다)."""
    K, D = default_camera_matrix_distortion()
    model = SphericalWindshieldModel(K, D, _MODEL, DEFAULT_SPHERE_CENTER, DEFAULT_SPHERE_RADIUS)
    before = model.project_point(0.1, 0.05, 0.6)

    runtime = RuntimeWindshieldProjector(model, IMG_W, IMG_H)
    runtime.project_points(_sample_points_at_distance(5.0, 20))
    runtime.unproject_pixels(_sample_pixels(20))

    after = model.project_point(0.1, 0.05, 0.6)
    assert before == after


def test_runtime_projector_rejects_degenerate_lut_size():
    K, D = default_camera_matrix_distortion()
    model = BaselineWindshieldModel(K, D, _MODEL)
    with pytest.raises(ValueError):
        RuntimeWindshieldProjector(model, IMG_W, IMG_H, lut_rows=1, lut_cols=1)


def test_build_runtime_projector_reuses_existing_exact_entry_point():
    """`build_runtime_projector()`는 새 재구성 로직을 만들지 않고 기존
    `build_projector()`를 그대로 재사용해야 한다(사용자 스펙 - exact entry
    point는 하나만 유지, 별도 API를 추가할 뿐 대체하지 않는다)."""
    K, D = default_camera_matrix_distortion()
    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.BASELINE,
        base_model_name=_MODEL,
        base_camera_matrix=K,
        base_distortion=D,
        success=True,
    )
    runtime = build_runtime_projector(result, IMG_W, IMG_H)
    assert isinstance(runtime.exact_model, BaselineWindshieldModel)


def test_runtime_projector_batch_output_shape_matches_input_count():
    K, D = default_camera_matrix_distortion()
    model = BaselineWindshieldModel(K, D, _MODEL)
    runtime = RuntimeWindshieldProjector(model, IMG_W, IMG_H)

    points = _sample_points_at_distance(5.0, 37)
    pixels = _sample_pixels(53)
    assert runtime.project_points(points).shape == (37, 2)
    assert runtime.unproject_pixels(pixels).shape == (53, 3)


def test_project_points_exact_batch_matches_scalar_loop_exactly_for_baseline():
    """Baseline은 진짜 벡터화된 경로(project_points_batch)를 타므로,
    scalar 반복 호출과 수치적으로 완전히 같아야 한다(근사가 아니라 exact
    batch 티어이므로 오차가 전혀 없어야 함)."""
    K, D = default_camera_matrix_distortion()
    model = BaselineWindshieldModel(K, D, _MODEL)
    points = _sample_points_at_distance(5.0, 25)

    scalar = np.array([model.project_point(*p) for p in points])
    batch = project_points_exact_batch(model, points)
    assert batch.shape == scalar.shape
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_project_points_exact_batch_matches_scalar_loop_exactly_for_spherical():
    """Spherical은 root-solve라 진짜 벡터화하지 않지만(반복 호출), 결과는
    scalar 호출과 완전히 동일해야 한다(exact - 근사 아님)."""
    K, D = default_camera_matrix_distortion()
    model = SphericalWindshieldModel(K, D, _MODEL, DEFAULT_SPHERE_CENTER, DEFAULT_SPHERE_RADIUS)
    points = _sample_points_at_distance(5.0, 10)

    scalar = np.array([model.project_point(*p) for p in points])
    batch = project_points_exact_batch(model, points)
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_unproject_pixels_exact_batch_matches_scalar_loop_exactly():
    K, D = default_camera_matrix_distortion()
    model = SphericalWindshieldModel(K, D, _MODEL, DEFAULT_SPHERE_CENTER, DEFAULT_SPHERE_RADIUS)
    pixels = _sample_pixels(15)

    scalar = np.array([model.unproject_pixel(*uv) for uv in pixels])
    batch = unproject_pixels_exact_batch(model, pixels)
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-10)
