"""
tests/test_windshield_residual_common.py
=============================================

calibration/windshield/residual_common.py 검증 - Grid와 RBF(STEP 3-B)가
공유하는 순수 유틸리티만 다룬다.

compute_ray_stability_deg는 이번 라운드에 시그니처를 일반화했다
(raw grid 배열 리스트 -> WindshieldModel 인스턴스 리스트) - 그래서 Grid와
RBF를 섞은 리스트를 넘겨도 동작해야 한다(둘 다 WindshieldModel API만
만족하면 되므로). 이 파일에서는 실제 RBF 모델까지는 만들지 않고, Grid
모델(ResidualRayWindshieldModel)만으로 시그니처/수학 자체를 검증한다 -
Grid+RBF 혼합 검증은 tests/test_windshield_residual_rbf_stabilization.py의
Test J류 테스트가 담당한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.windshield.residual_common import (
    compute_ray_stability_deg,
    fixed_evaluation_pixels,
    normalize_pixel_coordinates,
)
from calibration.windshield.residual_ray import ResidualRayWindshieldModel
from tests._windshield_test_utils import IMG_H, IMG_W, default_camera_matrix_distortion

_MODEL = CameraModelType.BROWN_CONRADY


def _grid_model(K, D, grid: np.ndarray) -> ResidualRayWindshieldModel:
    return ResidualRayWindshieldModel(K, D, _MODEL, grid, IMG_W, IMG_H)


# ---------------------------------------------------------------------------
# normalize_pixel_coordinates
# ---------------------------------------------------------------------------

def test_normalize_pixel_coordinates_maps_corners_to_unit_square():
    assert normalize_pixel_coordinates(0.0, 0.0, IMG_W, IMG_H) == pytest.approx((-1.0, -1.0))
    assert normalize_pixel_coordinates(IMG_W, IMG_H, IMG_W, IMG_H) == pytest.approx((1.0, 1.0))
    assert normalize_pixel_coordinates(IMG_W / 2.0, IMG_H / 2.0, IMG_W, IMG_H) == pytest.approx((0.0, 0.0))


# ---------------------------------------------------------------------------
# fixed_evaluation_pixels
# ---------------------------------------------------------------------------

def test_fixed_evaluation_pixels_is_deterministic_and_covers_image_bounds():
    pixels_a = fixed_evaluation_pixels(IMG_W, IMG_H, sample_rows=6, sample_cols=10)
    pixels_b = fixed_evaluation_pixels(IMG_W, IMG_H, sample_rows=6, sample_cols=10)

    assert pixels_a.shape == (60, 2)
    assert np.array_equal(pixels_a, pixels_b)
    assert pixels_a[:, 0].min() == pytest.approx(0.0)
    assert pixels_a[:, 0].max() == pytest.approx(IMG_W)
    assert pixels_a[:, 1].min() == pytest.approx(0.0)
    assert pixels_a[:, 1].max() == pytest.approx(IMG_H)


# ---------------------------------------------------------------------------
# compute_ray_stability_deg (일반화된 시그니처: models: list[WindshieldModel])
# ---------------------------------------------------------------------------

def test_ray_stability_is_zero_for_identical_models():
    K, D = default_camera_matrix_distortion()
    grid = np.random.default_rng(0).normal(scale=0.01, size=(6, 8, 3))
    model_a = _grid_model(K, D, grid)
    model_b = _grid_model(K, D, grid.copy())

    mean_deg, p95_deg = compute_ray_stability_deg([model_a, model_b], IMG_W, IMG_H)

    assert mean_deg is not None and p95_deg is not None
    assert mean_deg == pytest.approx(0.0, abs=1e-4)
    assert p95_deg == pytest.approx(0.0, abs=1e-4)


def test_ray_stability_is_positive_for_clearly_different_models():
    K, D = default_camera_matrix_distortion()
    rng = np.random.default_rng(1)
    grid_a = rng.normal(scale=0.005, size=(6, 8, 3))
    grid_b = grid_a + 0.05  # 명백히 큰 차이

    mean_deg, p95_deg = compute_ray_stability_deg([_grid_model(K, D, grid_a), _grid_model(K, D, grid_b)], IMG_W, IMG_H)

    assert mean_deg is not None and mean_deg > 0.1
    assert p95_deg is not None and p95_deg >= mean_deg


def test_ray_stability_returns_none_with_fewer_than_two_models():
    K, D = default_camera_matrix_distortion()
    grid = np.zeros((6, 8, 3))
    mean_deg, p95_deg = compute_ray_stability_deg([_grid_model(K, D, grid)], IMG_W, IMG_H)
    assert mean_deg is None and p95_deg is None


def test_ray_stability_is_resolution_independent_across_grid_shapes():
    """서로 다른 grid_rows/grid_cols 후보(3x4 대 6x8)로 만든 모델을 비교해도
    crash하지 않고 유한한 결과를 내야 한다 - 고정된 평가 샘플 픽셀(사용자
    스펙 23/24번)이 모델 자신의 내부 해상도가 아니라 이미지 좌표에서
    정의되기 때문에 가능하다."""
    K, D = default_camera_matrix_distortion()
    grid_small = np.random.default_rng(2).normal(scale=0.01, size=(3, 4, 3))
    grid_large = np.random.default_rng(3).normal(scale=0.01, size=(6, 8, 3))

    mean_deg, p95_deg = compute_ray_stability_deg(
        [_grid_model(K, D, grid_small), _grid_model(K, D, grid_large)], IMG_W, IMG_H,
    )

    assert mean_deg is not None and np.isfinite(mean_deg)
    assert p95_deg is not None and np.isfinite(p95_deg)


def test_ray_stability_sampling_is_deterministic_given_same_sample_grid():
    K, D = default_camera_matrix_distortion()
    grid_a = np.random.default_rng(4).normal(scale=0.01, size=(3, 4, 3))
    grid_b = np.random.default_rng(5).normal(scale=0.01, size=(8, 12, 3))
    models = [_grid_model(K, D, grid_a), _grid_model(K, D, grid_b)]

    mean_1, _ = compute_ray_stability_deg(models, IMG_W, IMG_H, sample_rows=6, sample_cols=10)
    mean_2, _ = compute_ray_stability_deg(models, IMG_W, IMG_H, sample_rows=6, sample_cols=10)

    assert mean_1 == pytest.approx(mean_2)
