"""
tests/test_windshield_residual_ray.py
==========================================

Phase 3-A(Residual Ray Grid) 검증 - 전부 Synthetic Dataset 기준이다.

중요: 이 테스트들은 "알고리즘이 설계대로 동작하는가"만 확인한다. 실차
windshield에서 이 모델이 Spherical보다 낫다/못하다는 결론은 여기서 절대
내리지 않는다(실차 Dataset 확보 전까지는 판단 보류).

실험적으로 확인된 사실(주석마다 재사용) - Standard solvePnP는 이미지 전역에
걸친 매끄러운(낮은 차수) 왜곡 필드의 상당 부분을 프레임별 rigid pose로
흡수해버린다(Spherical STEP 2의 "uniform shift가 거의 흡수됨" 교훈과 동일한
현상). 그래서 사용자 스펙 25번이 제시한 저차 다항식 GT field는 Baseline
대비 Residual Grid가 극적으로 개선되지는 않는다(테스트로 재현: 약 30% Hold-out
RMS 감소) - 이는 버그가 아니라, "이미지 전역에 걸친 매끈한 왜곡은 pose가 잘
설명해버리고, pose가 설명 못 하는 나머지만 grid가 잡아낸다"는 예상된 현상이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel, calibrate_baseline
from calibration.windshield.projection import build_projector
from calibration.windshield.residual_ray import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    ResidualRayWindshieldModel,
    bilinear_interpolate_grid,
    calibrate_residual_ray,
)
from export.windshield import export_windshield_yaml, windshield_model_from_yaml
from tests._windshield_test_utils import (
    build_synthetic_residual_ray_dataset,
    build_synthetic_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
    default_residual_delta_fn,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _config(K, D, **kwargs) -> WindshieldConfig:
    return WindshieldConfig(
        base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.RESIDUAL_RAY, **kwargs,
    )


# ---------------------------------------------------------------------------
# Bilinear interpolation - image center / exact node / cell center / border / corner
# ---------------------------------------------------------------------------

def _grid_3x3_with_known_corners():
    grid = np.zeros((3, 3, 3))
    grid[0, 0] = [1.0, 0.0, 0.0]
    grid[0, 2] = [0.0, 1.0, 0.0]
    grid[2, 0] = [0.0, 0.0, 1.0]
    grid[2, 2] = [1.0, 1.0, 1.0]
    grid[1, 1] = [9.0, 9.0, 9.0]  # 이미지 정중앙 node
    return grid


def test_bilinear_at_exact_grid_node_returns_node_value():
    grid = _grid_3x3_with_known_corners()
    assert bilinear_interpolate_grid(grid, 0.0, 0.0, 100.0, 100.0) == pytest.approx([1.0, 0.0, 0.0])
    assert bilinear_interpolate_grid(grid, 100.0, 100.0, 100.0, 100.0) == pytest.approx([1.0, 1.0, 1.0])


def test_bilinear_at_image_center_returns_center_node():
    grid = _grid_3x3_with_known_corners()
    assert bilinear_interpolate_grid(grid, 50.0, 50.0, 100.0, 100.0) == pytest.approx([9.0, 9.0, 9.0])


def test_bilinear_at_cell_center_is_average_of_four_nodes():
    grid = _grid_3x3_with_known_corners()
    # (25,25)는 node(0,0)=[1,0,0], (0,1)=[0,0,0], (1,0)=[0,0,0], (1,1)=[9,9,9]의 정중앙.
    expected = 0.25 * (np.array([1, 0, 0]) + np.array([0, 0, 0]) + np.array([0, 0, 0]) + np.array([9, 9, 9]))
    assert bilinear_interpolate_grid(grid, 25.0, 25.0, 100.0, 100.0) == pytest.approx(expected)


def test_bilinear_at_image_border_is_stable():
    grid = _grid_3x3_with_known_corners()
    # 왼쪽 경계(u=0) 중간 - node(0,0)=[1,0,0]과 node(2,0)=[0,0,1] 사이, node(1,0)=[0,0,0] 통과.
    v = bilinear_interpolate_grid(grid, 0.0, 50.0, 100.0, 100.0)
    assert v == pytest.approx([0.0, 0.0, 0.0])  # 정확히 node(1,0) 위치


def test_bilinear_at_image_corners():
    grid = _grid_3x3_with_known_corners()
    assert bilinear_interpolate_grid(grid, 0.0, 100.0, 100.0, 100.0) == pytest.approx([0.0, 0.0, 1.0])
    assert bilinear_interpolate_grid(grid, 100.0, 0.0, 100.0, 100.0) == pytest.approx([0.0, 1.0, 0.0])


def test_bilinear_clamps_out_of_bounds_pixels_without_crashing():
    grid = _grid_3x3_with_known_corners()
    assert bilinear_interpolate_grid(grid, -50.0, -50.0, 100.0, 100.0) == pytest.approx([1.0, 0.0, 0.0])
    assert bilinear_interpolate_grid(grid, 500.0, 500.0, 100.0, 100.0) == pytest.approx([1.0, 1.0, 1.0])


def test_bilinear_rejects_grid_too_small():
    with pytest.raises(ValueError):
        bilinear_interpolate_grid(np.zeros((1, 3, 3)), 0.0, 0.0, 100.0, 100.0)


# ---------------------------------------------------------------------------
# Zero-grid identity - 보정이 없으면 Baseline과 정확히 같아야 한다
# ---------------------------------------------------------------------------

def test_zero_grid_reproduces_baseline_exactly():
    K, D = default_camera_matrix_distortion()
    zero_grid = np.zeros((DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS, 3))
    model = ResidualRayWindshieldModel(K, D, _MODEL, zero_grid, 1280, 800)
    baseline = BaselineWindshieldModel(K, D, _MODEL)

    assert model.project_point(0.2, 0.15, 5.0) == pytest.approx(baseline.project_point(0.2, 0.15, 5.0), abs=1e-6)
    assert model.unproject_pixel(700.0, 380.0) == pytest.approx(baseline.unproject_pixel(700.0, 380.0), abs=1e-6)


# ---------------------------------------------------------------------------
# Calibration - K,D immutable / Train-Test / leakage / 실패 처리
# ---------------------------------------------------------------------------

def test_calibrate_residual_ray_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=1)

    calibrate_residual_ray(dataset, config, camera_config, train_ids, test_ids)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_calibrate_residual_ray_too_few_frames_returns_failure_not_exception():
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)

    result = calibrate_residual_ray(dataset, config, camera_config, [], [])

    assert result.success is False
    assert result.error_message


def test_calibrate_residual_ray_too_few_corners_for_grid_size_returns_failure():
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    # 큰 grid를 요구하면서 코너 수는 그대로 두면 "coverage 부족"으로 실패해야 한다.
    config = _config(K, D, residual_ray_hint={"grid_rows": 20, "grid_cols": 20})
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, [])

    assert result.success is False
    assert result.error_message


def test_residual_grid_fit_is_identical_regardless_of_test_data_content():
    """Spherical Test E와 동일한 패턴 - Test 코너를 크게 왜곡해도 fitted grid가
    (수치 허용오차 안에서) 동일해야 한다."""
    import copy

    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset_a = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    assert test_ids

    dataset_b = copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)

    config_a = _config(K, D)
    config_b = _config(K, D)
    result_a = calibrate_residual_ray(dataset_a, config_a, camera_config, train_ids, test_ids)
    result_b = calibrate_residual_ray(dataset_b, config_b, camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    rows, cols = DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS
    for r in range(rows):
        for c in range(cols):
            for axis in ("dx", "dy", "dz"):
                key = f"grid_{axis}_{r}_{c}"
                assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)
    assert result_a.test_residual_stats.rmse != pytest.approx(result_b.test_residual_stats.rmse, abs=1e-6)


# ---------------------------------------------------------------------------
# Synthetic GT recovery (섹션 26)
# ---------------------------------------------------------------------------

def test_zero_distortion_identity_grid_stays_near_zero():
    """왜곡이 전혀 없는 데이터셋에 fitting하면(사용자 스펙 26번 "zero-distortion
    identity"), grid가 (거의) 0으로 남아야 한다 - "고칠 게 없으면 고치지
    않는다"는 것을 확인."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)  # 굴절 없이 생성됨(baseline용)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, [])

    assert result.success
    grid_values = [
        result.fitted_params[f"grid_{axis}_{r}_{c}"]
        for r in range(DEFAULT_GRID_ROWS)
        for c in range(DEFAULT_GRID_COLS)
        for axis in ("dx", "dy", "dz")
    ]
    assert max(abs(v) for v in grid_values) < 0.01
    assert result.residual_stats.rmse < 0.05


def test_train_residual_decreases_vs_baseline():
    """Train RMS는 Baseline보다 확실히, 재현성 있게 개선돼야 한다 - 실험으로
    확인된 대로, 이 저차 다항식 GT는 pose가 상당 부분 흡수하므로 개선폭이
    극적이지는 않지만(약 30~40%) 모든 split에서 일관되게 나타난다."""
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    baseline_config = WindshieldConfig(base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D)
    baseline_result = calibrate_baseline(dataset, baseline_config, camera_config, train_ids, test_ids)

    grid_config = _config(K, D)
    grid_result = calibrate_residual_ray(dataset, grid_config, camera_config, train_ids, test_ids)

    assert grid_result.success
    assert grid_result.residual_stats.rmse < baseline_result.residual_stats.rmse * 0.85


def test_holdout_residual_improves_on_average_across_splits():
    """중요한 정직한 발견: 이 synthetic dataset은 프레임이 8장뿐이라(train 6/
    test 2), 개별 split 하나만 보면 Hold-out에서 Residual Grid가 Baseline보다
    "더 나쁘게" 나오는 경우도 실제로 있다(직접 확인됨 - test 프레임 2장짜리
    hold-out은 표본이 너무 작아 noise에 취약하다. 144-파라미터 grid를 8프레임
    으로 피팅하는 것 자체가 overfitting 여지가 있다는, 사용자 스펙 20번이
    경고한 현상의 실제 사례다). 여러 split의 평균으로 보면 방향은 일관되게
    개선이다 - 단일 split이 아니라 평균으로 판단해서 이 noise에 흔들리지
    않게 한다.
    """
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()

    baseline_test_rmses = []
    grid_test_rmses = []
    for seed in range(6):
        train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=seed)
        if len(test_ids) < 2:
            continue
        baseline_config = WindshieldConfig(base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D)
        baseline_result = calibrate_baseline(dataset, baseline_config, camera_config, train_ids, test_ids)
        grid_config = _config(K, D)
        grid_result = calibrate_residual_ray(dataset, grid_config, camera_config, train_ids, test_ids)
        if not grid_result.success:
            continue
        if baseline_result.test_residual_stats and grid_result.test_residual_stats:
            baseline_test_rmses.append(baseline_result.test_residual_stats.rmse)
            grid_test_rmses.append(grid_result.test_residual_stats.rmse)

    assert len(baseline_test_rmses) >= 4
    assert np.mean(grid_test_rmses) < np.mean(baseline_test_rmses)


def test_fitted_grid_is_smooth_between_adjacent_nodes():
    """섹션 26 "field smoothness" - 인접 node 간 값 차이가 극단적으로 튀지
    않아야 한다(smoothness regularization이 실제로 걸려 있다는 증거)."""
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, [])
    assert result.success

    rows, cols = DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS
    grid = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            grid[r, c, 0] = result.fitted_params[f"grid_dx_{r}_{c}"]
            grid[r, c, 1] = result.fitted_params[f"grid_dy_{r}_{c}"]
            grid[r, c, 2] = result.fitted_params[f"grid_dz_{r}_{c}"]

    max_jump = 0.0
    for r in range(rows):
        for c in range(cols - 1):
            max_jump = max(max_jump, float(np.linalg.norm(grid[r, c] - grid[r, c + 1])))
    for r in range(rows - 1):
        for c in range(cols):
            max_jump = max(max_jump, float(np.linalg.norm(grid[r, c] - grid[r + 1, c])))

    assert max_jump < 0.5  # ray-direction 단위 - 인접 node가 이 정도로 튀면 규제가 안 걸린 것


def test_edge_region_nodes_remain_finite_and_bounded():
    """섹션 26 "edge region" - 코너 관측이 드문 이미지 가장자리 node도 발산하지
    않고 유한한(그리고 bounds 안의) 값으로 남아야 한다."""
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, [])
    assert result.success

    rows, cols = DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS
    for r in (0, rows - 1):
        for c in range(cols):
            for axis in ("dx", "dy", "dz"):
                v = result.fitted_params[f"grid_{axis}_{r}_{c}"]
                assert np.isfinite(v)
                assert abs(v) <= 5.0  # _fit_residual_grid의 bounds와 일치


# ---------------------------------------------------------------------------
# YAML export / runtime reconstruction
# ---------------------------------------------------------------------------

def test_export_and_reconstruct_residual_ray_model(tmp_path):
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, [])
    assert result.success

    path = str(tmp_path / "windshield_residual_ray.yml")
    export_windshield_yaml(result, camera_config, path)
    reconstructed = windshield_model_from_yaml(path)
    original = build_projector(result)

    test_point = (0.15, -0.1, 5.0)
    assert reconstructed.project_point(*test_point) == pytest.approx(original.project_point(*test_point), abs=1e-6)
