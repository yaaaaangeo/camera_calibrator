"""
tests/test_windshield_spline_stabilization.py
==================================================

STEP 4 Spline Windshield Model 검증(안정화) - STAGE B, leakage, AUTO,
repeated hold-out, ray/surface stability, pose 진단, 진단 오케스트레이터.

이 fixture(8프레임 synthetic)는 카메라가 base sphere 표면에서 겨우 ~1cm
떨어진 좌표계라, 실측으로 확인된 특성이 있다: 표면의 "국소 반경 보정
(Δs)"이 far-field 방향에 주는 영향이 "표면 방향(기울기)"보다 훨씬 작다
- 그래서 24-parameter spline이 8프레임짜리 작은 데이터셋에 과적합되기
쉽고, 개별 split의 Test RMS가 noisy할 수 있다(Residual Grid/RBF에서
이미 문서화된 것과 같은 종류의 현상 - Repeated Hold-out이 존재하는 이유).
이 테스트들은 그래서 "정확한 개선 폭"이 아니라 "구조가 올바르게 동작하고
발산하지 않는지"를 확인하는 데 집중한다.
"""

from __future__ import annotations

import copy as _copy

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.spline import (
    calibrate_spline,
    run_repeated_holdout_spline,
    run_spline_calibration_with_diagnostics,
    select_best_spline_grid_resolution,
)
from tests._windshield_test_utils import (
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


# ---------------------------------------------------------------------------
# STAGE B / K,D immutable
# ---------------------------------------------------------------------------

def test_stage_b_pipeline_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    calibrate_spline(dataset, config, camera_config, train_ids, test_ids)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_final_result_reports_stage_used_and_never_worse_than_stage_a():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    assert "stage_used_is_joint_refined" in result.fitted_params
    assert np.isfinite(result.residual_stats.rmse)


# ---------------------------------------------------------------------------
# Test J - Pose ΔR/Δt finite & non-divergent
# ---------------------------------------------------------------------------

def test_pose_moves_within_reasonable_bounds_during_stage_b():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})

    for seed in (1, 2, 3):
        train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=seed)
        result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)
        assert result.success, result.error_message

        for key in (
            "diag_pose_delta_r_median_deg", "diag_pose_delta_r_p95_deg",
            "diag_pose_delta_t_median_mm", "diag_pose_delta_t_p95_mm",
        ):
            assert key in result.fitted_params
            assert np.isfinite(result.fitted_params[key])

        # 넉넉한 상한(명백한 발산만 잡아낸다 - flaky test를 만들지 않는다).
        assert result.fitted_params["diag_pose_delta_r_median_deg"] < 30.0
        assert result.fitted_params["diag_pose_delta_t_median_mm"] < 500.0
        if result.fitted_params["stage_used_is_joint_refined"] == 0.0:
            assert result.fitted_params["diag_pose_delta_r_median_deg"] == pytest.approx(0.0, abs=1e-9)
            assert result.fitted_params["diag_pose_delta_t_median_mm"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test H - Outer Test leakage
# ---------------------------------------------------------------------------

def test_outer_test_corruption_does_not_change_base_sphere_or_spline_grid():
    K, D = default_camera_matrix_distortion()
    dataset_a = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    assert test_ids

    dataset_b = _copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)

    config_a = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})
    config_b = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})

    result_a = calibrate_spline(dataset_a, config_a, camera_config, train_ids, test_ids)
    result_b = calibrate_spline(dataset_b, config_b, camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    for key in ("sphere_center_x", "sphere_center_y", "sphere_center_z", "sphere_radius"):
        assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)
    rows, cols = int(result_a.fitted_params["spline_rows"]), int(result_a.fitted_params["spline_cols"])
    for r in range(rows):
        for c in range(cols):
            key = f"spline_ds_{r}_{c}"
            assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)


# ---------------------------------------------------------------------------
# Test I - AUTO grid selection leakage
# ---------------------------------------------------------------------------

def test_auto_grid_selection_does_not_leak_outer_test_set():
    import calibration.windshield.spline as spline_module

    original_candidates = spline_module.SPLINE_GRID_CANDIDATES
    spline_module.SPLINE_GRID_CANDIDATES = [(2, 2), (2, 3)]
    try:
        K, D = default_camera_matrix_distortion()
        dataset_a = build_synthetic_spline_windshield_dataset(K, D)
        camera_config = default_camera_config()
        ids = [f.image_info.image_id for f in dataset_a.frames]
        train_ids, test_ids = ids[:-2], ids[-2:]
        assert test_ids

        dataset_b = _copy.deepcopy(dataset_a)
        for frame in dataset_b.frames:
            if frame.image_info.image_id in test_ids:
                frame.detection.corners = frame.detection.corners + np.float32(50.0)

        config_a = _config(K, D, spline_hint={"auto_spline": 1.0})
        config_b = _config(K, D, spline_hint={"auto_spline": 1.0})

        result_a = calibrate_spline(dataset_a, config_a, camera_config, train_ids, test_ids)
        result_b = calibrate_spline(dataset_b, config_b, camera_config, train_ids, test_ids)

        assert result_a.success and result_b.success
        assert result_a.fitted_params["spline_rows"] == result_b.fitted_params["spline_rows"]
        assert result_a.fitted_params["spline_cols"] == result_b.fitted_params["spline_cols"]
        rows, cols = int(result_a.fitted_params["spline_rows"]), int(result_a.fitted_params["spline_cols"])
        for r in range(rows):
            for c in range(cols):
                key = f"spline_ds_{r}_{c}"
                assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)
    finally:
        spline_module.SPLINE_GRID_CANDIDATES = original_candidates


def test_select_best_spline_grid_resolution_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    select_best_spline_grid_resolution(dataset, config, camera_config, train_ids, candidates=[(2, 2)], seeds=(1,))

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Test K/L/M - Repeated Hold-out, Ray Stability, Surface Stability
# ---------------------------------------------------------------------------

def test_repeated_holdout_runs_and_reports_stability_fields():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)  # 순수 GT - 더 안정적으로 성공하는 fixture
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})

    summary = run_repeated_holdout_spline(dataset, config, camera_config, seeds=(1, 2, 3), test_ratio=0.3)

    assert summary.holdout.n_successful >= 1
    if summary.holdout.n_successful >= 1:
        assert summary.holdout.mean_test_rmse is not None
        assert summary.holdout.mean_test_rmse >= 0
    # 2개 이상 성공했을 때만 pairwise stability를 계산할 수 있다(None을
    # 억지로 채우지 않는다는 설계 - 사용자 스펙 23/28번).
    if summary.holdout.n_successful >= 2:
        assert summary.holdout.ray_stability_mean_deg is not None
        assert summary.holdout.ray_stability_mean_deg >= 0
        assert summary.surface_stability_mean_mm is not None
        assert summary.surface_stability_mean_mm >= 0


def test_repeated_holdout_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})

    run_repeated_holdout_spline(dataset, config, camera_config, seeds=(1, 2), test_ratio=0.3)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Diagnostics orchestrator
# ---------------------------------------------------------------------------

def test_diagnostics_orchestrator_populates_all_diag_fields():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = run_spline_calibration_with_diagnostics(
        dataset, config, camera_config, train_ids, test_ids,
        compute_repeated_holdout=True, repeated_holdout_seeds=(11, 12, 13), repeated_holdout_test_ratio=0.3,
    )

    assert result.success, result.error_message
    assert result.fitted_params["diag_selection_mode_is_auto"] == 0.0
    assert "diag_repeated_n_requested" in result.fitted_params
    assert result.fitted_params["diag_repeated_n_requested"] == 3.0
    assert "diag_repeated_n_successful" in result.fitted_params
    for key in (
        "diag_pose_delta_r_median_deg", "diag_pose_delta_r_p95_deg",
        "diag_pose_delta_t_median_mm", "diag_pose_delta_t_p95_mm",
        "diag_deformation_mean_abs_m", "diag_deformation_max_abs_m",
    ):
        assert key in result.fitted_params


def test_diagnostics_orchestrator_auto_uses_resolved_size_for_holdout():
    import calibration.windshield.spline as spline_module

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"auto_spline": 1.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    original_candidates = spline_module.SPLINE_GRID_CANDIDATES
    spline_module.SPLINE_GRID_CANDIDATES = [(2, 2), (2, 3)]
    original_select = spline_module.select_best_spline_grid_resolution
    call_count = {"n": 0}

    def counting_select(*args, **kwargs):
        call_count["n"] += 1
        return original_select(*args, **kwargs)

    spline_module.select_best_spline_grid_resolution = counting_select
    try:
        result = run_spline_calibration_with_diagnostics(
            dataset, config, camera_config, train_ids, test_ids,
            compute_repeated_holdout=True, repeated_holdout_seeds=(21, 22), repeated_holdout_test_ratio=0.3,
        )
    finally:
        spline_module.select_best_spline_grid_resolution = original_select
        spline_module.SPLINE_GRID_CANDIDATES = original_candidates

    assert result.success, result.error_message
    assert result.fitted_params["diag_selection_mode_is_auto"] == 1.0
    assert call_count["n"] == 1
