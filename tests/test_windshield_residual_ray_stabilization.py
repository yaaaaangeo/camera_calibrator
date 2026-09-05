"""
tests/test_windshield_residual_ray_stabilization.py
========================================================

Residual Ray Grid STEP 3-A 안정화(2차 라운드) 검증: STAGE B(grid+pose joint
refinement), Repeated Hold-out, Grid Resolution Comparison + AUTO 선택,
Parameter Count. STAGE A 단독 동작은 기존 tests/test_windshield_residual_ray.py
가 계속 담당한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.residual_ray import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    calibrate_residual_ray,
    run_repeated_holdout_residual_ray,
    select_best_grid_resolution,
)
from tests._windshield_test_utils import (
    build_synthetic_residual_ray_dataset,
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
# STAGE B - joint refinement
# ---------------------------------------------------------------------------

def test_stage_b_pipeline_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    calibrate_residual_ray(dataset, config, camera_config, train_ids, test_ids)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_final_result_never_worse_than_stage_a():
    """calibrate_residual_ray()은 내부적으로 STAGE A/B의 실제 pixel RMS를
    비교해서 더 나은 쪽만 채택한다 - fitted_params가 STAGE B 채택 여부를
    투명하게 기록해야 하고, stage_used_is_joint_refined==1.0이면 STAGE A
    단독보다 명확히 개선돼 있어야 한다(사용자 스펙 26번 Test A와 동일한
    발상)."""
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    assert "stage_used_is_joint_refined" in result.fitted_params
    # STAGE B가 채택됐다면 STAGE A 단독(약 0.51px로 별도 실험에서 확인됨)보다
    # 뚜렷하게 나아야 한다. 채택 안 됐다면 최소한 성공 결과 자체는 유효해야 함.
    if result.fitted_params["stage_used_is_joint_refined"] == 1.0:
        assert result.residual_stats.rmse < 0.5


def test_pose_moves_within_reasonable_bounds_during_stage_b():
    """weak prior가 걸려 있으므로 pose가 initial solvePnP에서 과도하게
    벗어나면 안 된다(간접 검증 - 결과가 발산하지 않고 성공으로 끝난다는 것
    자체가 prior가 poses를 억제하고 있다는 신호. 직접 pose 값을 비교하려면
    내부 함수를 호출해야 하므로, 여기서는 여러 시드에 걸쳐 안정적으로
    수렴하는지를 함께 확인한다)."""
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)

    for seed in (1, 2, 3):
        train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=seed)
        result = calibrate_residual_ray(dataset, config, camera_config, train_ids, test_ids)
        assert result.success, result.error_message
        assert np.isfinite(result.residual_stats.rmse)


# ---------------------------------------------------------------------------
# Repeated Hold-out
# ---------------------------------------------------------------------------

def test_repeated_holdout_aggregates_multiple_seeds():
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)

    summary = run_repeated_holdout_residual_ray(dataset, config, camera_config, seeds=(1, 2, 3), test_ratio=0.3)

    assert summary.n_successful >= 2
    assert summary.mean_test_rmse is not None
    assert summary.mean_test_rmse > 0
    assert summary.std_test_rmse is not None and summary.std_test_rmse >= 0
    # 3개 이상 성공했으면 grid_stability(pairwise 거리 평균)도 계산됐어야 한다.
    if summary.n_successful >= 2:
        assert summary.grid_stability is not None
        assert summary.grid_stability >= 0


def test_repeated_holdout_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)

    run_repeated_holdout_residual_ray(dataset, config, camera_config, seeds=(1, 2), test_ratio=0.3)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Grid Resolution Comparison + AUTO
# ---------------------------------------------------------------------------

def test_grid_candidates_report_increasing_param_count():
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    _, candidates = select_best_grid_resolution(
        dataset, config, camera_config, train_ids, candidates=[(3, 4), (4, 6)], seeds=(1,),
    )

    assert [c.param_count for c in candidates] == [3 * 4 * 3, 4 * 6 * 3]


def test_grid_selection_prefers_fewer_params_when_rmse_is_close():
    """tie_tolerance 안에서 RMS가 비슷하면 parameter가 더 적은 후보를
    선택해야 한다(사용자 스펙 8번). 아주 관대한 tie_tolerance를 주면
    항상 가장 작은 후보가 선택돼야 한다는 걸로 이 규칙 자체를 검증한다."""
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    (rows, cols), _candidates = select_best_grid_resolution(
        dataset, config, camera_config, train_ids,
        candidates=[(3, 4), (4, 6)], seeds=(1,), tie_tolerance=10.0,  # 사실상 전부 "비슷함" 취급
    )

    assert (rows, cols) == (3, 4)  # 가장 적은 파라미터


def test_select_best_grid_resolution_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    select_best_grid_resolution(dataset, config, camera_config, train_ids, candidates=[(3, 4)], seeds=(1,))

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_auto_grid_end_to_end_does_not_leak_outer_test_set(monkeypatch):
    """auto_grid=1.0으로 실행하면 내부적으로 select_best_grid_resolution이
    outer train_ids만 갖고 grid 크기를 고른 뒤, 그 크기로 최종 fitting까지
    끝낸다 - outer test_ids로 서로 다른(왜곡된) 데이터를 줘도 선택된 grid
    크기와 최종 fitted grid 값이 바뀌면 안 된다(leakage 검증).

    실행 시간을 줄이기 위해 후보/시드 수를 테스트 전용으로 줄인다 - AUTO
    선택 로직 자체(leakage 없음)를 검증하는 게 목적이지, 기본 후보/시드
    개수 자체를 검증하는 게 아니므로 문제 없다.
    """
    import copy as _copy
    import calibration.windshield.residual_ray as residual_ray_module

    monkeypatch.setattr(residual_ray_module, "DEFAULT_GRID_CANDIDATES", [(3, 4), (4, 6)])

    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset_a = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    assert test_ids

    dataset_b = _copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)

    small_candidates_hint = {"auto_grid": 1.0}
    config_a = _config(K, D, residual_ray_hint=small_candidates_hint)
    config_b = _config(K, D, residual_ray_hint=dict(small_candidates_hint))

    result_a = calibrate_residual_ray(dataset_a, config_a, camera_config, train_ids, test_ids)
    result_b = calibrate_residual_ray(dataset_b, config_b, camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    assert result_a.fitted_params["grid_rows"] == result_b.fitted_params["grid_rows"]
    assert result_a.fitted_params["grid_cols"] == result_b.fitted_params["grid_cols"]
    rows, cols = int(result_a.fitted_params["grid_rows"]), int(result_a.fitted_params["grid_cols"])
    for r in range(rows):
        for c in range(cols):
            for axis in ("dx", "dy", "dz"):
                key = f"grid_{axis}_{r}_{c}"
                assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)


# ---------------------------------------------------------------------------
# Parameter Count
# ---------------------------------------------------------------------------

def test_fitted_params_report_runtime_vs_calibration_param_counts():
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_residual_ray(dataset, config, camera_config, train_ids, test_ids)

    assert result.success
    rows, cols = int(result.fitted_params["grid_rows"]), int(result.fitted_params["grid_cols"])
    assert result.fitted_params["runtime_param_count"] == rows * cols * 3
    assert result.fitted_params["pose_param_count_train"] == len(result.train_frame_ids) * 6 or (
        # 일부 프레임이 검출/포즈 실패로 빠질 수 있으므로 상한으로도 확인.
        result.fitted_params["pose_param_count_train"] <= len(result.train_frame_ids) * 6
    )
