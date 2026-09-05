from __future__ import annotations

import copy

import numpy as np
import pytest

from calibration.types import CameraModelType, Dataset
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.projection import build_projector
from calibration.windshield.residual_common import compute_ray_stability_deg
from calibration.windshield.residual_ray import calibrate_residual_ray
from calibration.windshield.residual_rbf import (
    ResidualRBFWindshieldModel,
    calibrate_residual_rbf,
    run_repeated_holdout_residual_rbf,
    run_residual_rbf_calibration_with_diagnostics,
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
        base_model_name=_MODEL,
        base_camera_matrix=K,
        base_distortion=D,
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        **kwargs,
    )


def _rbf_config(K, D, hint=None) -> WindshieldConfig:
    return _config(
        K,
        D,
        residual_ray_hint=hint or {"method": "rbf", "rbf_num_centers": 8.0, "rbf_smoothing": 1e-4},
    )


def _dataset_pair_with_shifted_test_corners(K, D):
    dataset_a = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    dataset_b = copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)
    return dataset_a, dataset_b, train_ids, test_ids


def _assert_same_rbf_training_representation(result_a, result_b):
    for key in ("rbf_num_centers", "rbf_smoothing"):
        assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-12)
    n = int(result_a.fitted_params["rbf_num_centers"])
    assert int(result_b.fitted_params["rbf_num_centers"]) == n
    for i in range(n):
        for key in (
            f"rbf_center_u_{i}",
            f"rbf_center_v_{i}",
            f"rbf_residual_dx_{i}",
            f"rbf_residual_dy_{i}",
            f"rbf_residual_dz_{i}",
        ):
            assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)


def test_outer_test_plus_50px_does_not_change_rbf_training_representation():
    K, D = default_camera_matrix_distortion()
    dataset_a, dataset_b, train_ids, test_ids = _dataset_pair_with_shifted_test_corners(K, D)
    camera_config = default_camera_config()

    result_a = calibrate_residual_rbf(dataset_a, _rbf_config(K, D), camera_config, train_ids, test_ids)
    result_b = calibrate_residual_rbf(dataset_b, _rbf_config(K, D), camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    _assert_same_rbf_training_representation(result_a, result_b)
    assert result_a.test_residual_stats.rmse != pytest.approx(result_b.test_residual_stats.rmse, abs=1e-6)


def test_auto_rbf_selection_does_not_leak_outer_test_set(monkeypatch):
    import calibration.windshield.residual_rbf as residual_rbf_module

    monkeypatch.setattr(residual_rbf_module, "RBF_CENTER_CANDIDATES", [8, 16])
    monkeypatch.setattr(residual_rbf_module, "RBF_SMOOTHING_CANDIDATES", [1e-4, 1e-3])
    monkeypatch.setattr(residual_rbf_module, "DEFAULT_REPEATED_HOLDOUT_SEEDS", (1, 2))

    K, D = default_camera_matrix_distortion()
    dataset_a, dataset_b, train_ids, test_ids = _dataset_pair_with_shifted_test_corners(K, D)
    camera_config = default_camera_config()
    hint = {"method": "rbf", "auto_rbf": 1.0}

    result_a = calibrate_residual_rbf(dataset_a, _rbf_config(K, D, dict(hint)), camera_config, train_ids, test_ids)
    result_b = calibrate_residual_rbf(dataset_b, _rbf_config(K, D, dict(hint)), camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    _assert_same_rbf_training_representation(result_a, result_b)


def test_rbf_stage_b_pose_diagnostics_are_finite_and_bounded():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_residual_rbf(dataset, _rbf_config(K, D), camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    for key in (
        "diag_pose_delta_r_median_deg",
        "diag_pose_delta_r_p95_deg",
        "diag_pose_delta_t_median_mm",
        "diag_pose_delta_t_p95_mm",
    ):
        assert key in result.fitted_params
        assert np.isfinite(result.fitted_params[key])
    assert result.fitted_params["diag_pose_delta_r_p95_deg"] < 30.0
    assert result.fitted_params["diag_pose_delta_t_p95_mm"] < 500.0
    if result.fitted_params["stage_used_is_joint_refined"] == 0.0:
        assert result.fitted_params["diag_pose_delta_r_median_deg"] == pytest.approx(0.0, abs=1e-9)
        assert result.fitted_params["diag_pose_delta_t_median_mm"] == pytest.approx(0.0, abs=1e-9)


def test_rbf_repeated_holdout_runs_and_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()

    summary = run_repeated_holdout_residual_rbf(
        dataset, _rbf_config(K, D), camera_config, seeds=(1, 2, 3), test_ratio=0.3
    )

    assert summary.n_successful >= 2
    assert summary.mean_test_rmse is not None and np.isfinite(summary.mean_test_rmse)
    assert summary.std_test_rmse is not None and summary.std_test_rmse >= 0
    if summary.mean_test_p95 is not None:
        assert np.isfinite(summary.mean_test_p95)
    if summary.mean_edge_rms is not None:
        assert np.isfinite(summary.mean_edge_rms)
    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_rbf_angular_stability_is_reported_for_multiple_models():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()

    summary = run_repeated_holdout_residual_rbf(
        dataset, _rbf_config(K, D), camera_config, seeds=(1, 2, 3), test_ratio=0.3
    )

    assert summary.ray_stability_mean_deg is not None
    assert summary.ray_stability_p95_deg is not None
    assert np.isfinite(summary.ray_stability_mean_deg)
    assert np.isfinite(summary.ray_stability_p95_deg)
    assert summary.ray_stability_mean_deg >= 0
    assert summary.ray_stability_p95_deg >= summary.ray_stability_mean_deg - 1e-12

    centers = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    values = np.zeros((4, 3), dtype=np.float64)
    model = ResidualRBFWindshieldModel(K, D, _MODEL, centers, values, 1280, 800, smoothing=0.0)
    mean, p95 = compute_ray_stability_deg([model, model], 1280, 800)
    assert mean == pytest.approx(0.0, abs=1e-6)
    assert p95 == pytest.approx(0.0, abs=1e-6)


def test_grid_and_rbf_use_the_same_outer_split_for_fair_comparison():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=4)

    grid_result = calibrate_residual_ray(dataset, _config(K, D), camera_config, train_ids, test_ids)
    rbf_result = calibrate_residual_rbf(dataset, _rbf_config(K, D), camera_config, train_ids, test_ids)

    assert grid_result.success and rbf_result.success
    assert grid_result.train_frame_ids == rbf_result.train_frame_ids == train_ids
    assert grid_result.test_frame_ids == rbf_result.test_frame_ids == test_ids


def test_same_synthetic_gt_grid_and_rbf_both_produce_valid_runtime_models():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=2)

    grid_result = calibrate_residual_ray(dataset, _config(K, D), camera_config, train_ids, test_ids)
    rbf_result = calibrate_residual_rbf(dataset, _rbf_config(K, D), camera_config, train_ids, test_ids)

    assert grid_result.success and rbf_result.success
    assert grid_result.test_residual_stats and np.isfinite(grid_result.test_residual_stats.rmse)
    assert rbf_result.test_residual_stats and np.isfinite(rbf_result.test_residual_stats.rmse)

    for model in (build_projector(grid_result), build_projector(rbf_result)):
        ray = model.unproject_pixel(700.0, 380.0)
        uv = model.project_point(0.15, -0.1, 5.0)
        assert np.all(np.isfinite(ray))
        assert np.all(np.isfinite(uv))


def test_rbf_diagnostics_repeated_holdout_uses_outer_train_subset_only(monkeypatch):
    import calibration.windshield.residual_rbf as residual_rbf_module

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    seen_ids = []

    def fake_repeated(inner_dataset, config, camera_config_arg, seeds, test_ratio):
        seen_ids.extend(f.image_info.image_id for f in inner_dataset.frames)
        from calibration.windshield.residual_common import RepeatedHoldoutSummary

        return RepeatedHoldoutSummary(seeds_used=list(seeds), n_successful=0)

    monkeypatch.setattr(residual_rbf_module, "run_repeated_holdout_residual_rbf", fake_repeated)

    result = run_residual_rbf_calibration_with_diagnostics(
        dataset,
        _rbf_config(K, D),
        camera_config,
        train_ids,
        test_ids,
        compute_repeated_holdout=True,
        repeated_holdout_seeds=(1,),
    )

    assert result.success, result.error_message
    assert seen_ids
    assert set(seen_ids) == set(train_ids)
    assert not (set(seen_ids) & set(test_ids))
