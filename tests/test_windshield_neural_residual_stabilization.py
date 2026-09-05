"""
tests/test_windshield_neural_residual_stabilization.py
==============================================================

STEP 5 - Neural Residual Windshield Model 안정화 검증 (Test F-P + leakage
매핑): Outer Test leakage, Early stopping, Best checkpoint, Seed
reproducibility/stability, STAGE B pose, Repeated Hold-out, Ray Stability,
Grid/RBF/Neural 공정 비교(같은 split, 같은 synthetic GT).

PyTorch가 없는 환경에서는 이 파일 전체를 skip한다.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402 - importorskip 이후에만 안전하게 import

from calibration.types import CameraModelType, Dataset
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.neural_residual import (
    DEFAULT_NEURAL_HIDDEN_DIMS,
    _decode_state_dict,
    _fit_neural_stage_a,
    _neural_settings,
    _split_train_validation,
    _train_mlp,
    calibrate_neural_residual,
    compute_seed_stability_deg,
    run_neural_residual_calibration_with_diagnostics,
    run_repeated_holdout_neural_residual,
)
from calibration.windshield.projection import build_projector
from calibration.windshield.refraction import normalize
from calibration.windshield.residual_common import collect_corner_arrays, compute_ray_stability_deg, normalize_pixel_coordinates
from calibration.windshield.residual_rbf import calibrate_residual_rbf
from calibration.windshield.residual_ray import calibrate_residual_ray
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
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


def _neural_config(K, D, hint=None) -> WindshieldConfig:
    base_hint = {"method": "neural", "neural_max_epochs": 80, "neural_patience": 15, "neural_seed": 42}
    return _config(K, D, residual_ray_hint={**base_hint, **(hint or {})})


def _dataset_pair_with_shifted_test_corners(K, D):
    dataset_a = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    dataset_b = copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)
    return dataset_a, dataset_b, train_ids, test_ids


# ---------------------------------------------------------------------------
# Test F/L - Outer Test +50px leakage: 학습된 state_dict/architecture가
# Outer Test 코너 왜곡과 무관하게 동일해야 한다(같은 seed 기준).
# ---------------------------------------------------------------------------

def test_outer_test_corruption_does_not_change_trained_weights():
    """Test A(사용자 스펙 5-A번) - 예측값 몇 점 비교로 끝내지 않고, 학습된
    state_dict의 모든 tensor를 직접(key별로) 비교한다. 같은 seed + CPU
    deterministic path이므로 완전히 동일해야 한다(rtol=0, 아주 작은 atol만
    부동소수점 유도 오차를 흡수)."""
    K, D = default_camera_matrix_distortion()
    dataset_a, dataset_b, train_ids, test_ids = _dataset_pair_with_shifted_test_corners(K, D)
    camera_config = default_camera_config()

    result_a = calibrate_neural_residual(dataset_a, _neural_config(K, D), camera_config, train_ids, test_ids)
    result_b = calibrate_neural_residual(dataset_b, _neural_config(K, D), camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    for key in (
        "neural_num_hidden_layers", "neural_activation_code", "neural_best_epoch",
        "neural_best_train_ray_loss", "neural_best_val_ray_loss",
        "neural_best_train_total_loss", "neural_best_val_total_loss",
        "stage_used_is_joint_refined",
    ):
        assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)

    state_a = _decode_state_dict(result_a.neural_state_dict_b64)
    state_b = _decode_state_dict(result_b.neural_state_dict_b64)
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=1e-6)

    model_a = build_projector(result_a)
    model_b = build_projector(result_b)
    for u, v in ((640.0, 400.0), (300.0, 200.0), (900.0, 600.0)):
        assert model_a.unproject_pixel(u, v) == pytest.approx(model_b.unproject_pixel(u, v), abs=1e-5)

    # Test 쪽 평가 지표(RMS)는 왜곡된 코너 때문에 실제로 달라져야 한다 -
    # "weight가 동일한데 왜 이걸 확인하냐"가 아니라 "leakage가 없다는 것"과
    # "corruption이 실제로 관측 가능한 효과를 낸다는 것"을 함께 확인한다.
    assert result_a.test_residual_stats.rmse != pytest.approx(result_b.test_residual_stats.rmse, abs=1e-6)


# ---------------------------------------------------------------------------
# Test G - Early stopping: patience를 작게 주면 max_epochs 전에 멈춘다.
# ---------------------------------------------------------------------------

def test_early_stopping_triggers_before_max_epochs_with_small_patience():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_frames = dataset.frames[:-2]

    ok_frames, rvecs, tvecs, _failed = solve_poses_fixed_intrinsics(train_frames, K, D, _MODEL)
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(ok_frames, rvecs, tvecs, baseline)
    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    width, height = 1280.0, 800.0
    config = _neural_config(K, D, {"neural_max_epochs": 2000, "neural_patience": 3})
    settings = _neural_settings(config)

    outcome = _fit_neural_stage_a(observed_pixels_arr, d_obs_arr, p_cam_arr, width, height, settings)

    assert outcome.stopped_early is True
    assert outcome.best_epoch < settings.max_epochs - settings.patience


# ---------------------------------------------------------------------------
# Test H - Best checkpoint: 마지막 epoch가 아니라 best validation epoch의
# weight가 최종으로 쓰였는지 - stopped_early가 True면 best_epoch는 항상
# 실제로 학습이 멈춘 epoch(best_epoch + patience 근방)보다 앞서야 한다.
# ---------------------------------------------------------------------------

def test_best_checkpoint_is_not_necessarily_the_last_epoch():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_frames = dataset.frames[:-2]

    ok_frames, rvecs, tvecs, _failed = solve_poses_fixed_intrinsics(train_frames, K, D, _MODEL)
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(ok_frames, rvecs, tvecs, baseline)
    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    config = _neural_config(K, D, {"neural_max_epochs": 300, "neural_patience": 5})
    settings = _neural_settings(config)
    outcome = _fit_neural_stage_a(observed_pixels_arr, d_obs_arr, p_cam_arr, 1280.0, 800.0, settings)

    # stopped_early가 아니라면(300 epoch 내내 계속 개선) best_epoch가 마지막
    # epoch여도 정상이다 - 이 테스트는 "만약 조기 종료됐다면 best != last"라는
    # 계약만 확인한다(flaky한 절대 epoch 수 assertion을 피한다).
    if outcome.stopped_early:
        assert outcome.best_epoch < settings.max_epochs - 1
    assert np.isfinite(outcome.best_val_ray_loss)


def test_best_checkpoint_matches_argmin_of_validation_history_and_exact_weights():
    """Test H(사용자 스펙 5-B번, 강화) - 두 가지를 직접 증명한다:

    1. `outcome.best_epoch`가 진짜로 매 epoch validation ray loss 기록의
       argmin과 일치하는지(`history` debug hook으로 직접 기록).
    2. 반환된 `state_dict`가 정말 그 epoch의 weight인지 - 같은 seed/데이터로
       정확히 `best_epoch+1` epoch만(그 사이에 조기 종료가 끼어들지 못하게
       patience를 크게 줘서) 다시 학습했을 때 나오는 state_dict와 완전히
       (bit-level로) 같아야 한다. CPU 결정론적 경로이므로 정확히 재현된다.
    """
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    train_frames = dataset.frames[:-2]

    ok_frames, rvecs, tvecs, _failed = solve_poses_fixed_intrinsics(train_frames, K, D, _MODEL)
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    observed_pixels_per_frame, d_obs_per_frame, p_cam_per_frame = collect_corner_arrays(ok_frames, rvecs, tvecs, baseline)
    observed_pixels_arr = np.concatenate(observed_pixels_per_frame, axis=0)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    config = _neural_config(K, D, {"neural_max_epochs": 150, "neural_patience": 12})
    settings = _neural_settings(config)
    history: list[float] = []
    outcome = _fit_neural_stage_a(observed_pixels_arr, d_obs_arr, p_cam_arr, 1280.0, 800.0, settings, history=history)

    assert len(history) >= 1
    assert outcome.best_epoch == int(np.argmin(history))
    assert outcome.best_val_ray_loss == pytest.approx(history[outcome.best_epoch], abs=1e-9)

    # 2. 같은 지점까지 재학습해서 나온 weight가 정확히 같은지 직접 비교.
    truncated_settings = dataclasses.replace(settings, patience=10**6)
    n = observed_pixels_arr.shape[0]
    train_idx, val_idx = _split_train_validation(n, settings.validation_ratio, settings.seed)
    target_dirs = np.array([normalize(p) for p in p_cam_arr])
    uv_norm = np.array([normalize_pixel_coordinates(u, v, 1280.0, 800.0) for u, v in observed_pixels_arr])
    truncated_outcome = _train_mlp(
        uv_norm[train_idx], d_obs_arr[train_idx], target_dirs[train_idx],
        uv_norm[val_idx], d_obs_arr[val_idx], target_dirs[val_idx],
        truncated_settings, max_epochs=outcome.best_epoch + 1,
    )

    assert truncated_outcome.state_dict.keys() == outcome.state_dict.keys()
    for key in outcome.state_dict:
        torch.testing.assert_close(outcome.state_dict[key], truncated_outcome.state_dict[key], rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test I - Fixed Seed Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_split_produces_reproducible_model():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]
    config = _neural_config(K, D, {"neural_seed": 7})

    result_1 = calibrate_neural_residual(dataset, config, camera_config, train_ids, test_ids)
    result_2 = calibrate_neural_residual(dataset, config, camera_config, train_ids, test_ids)

    assert result_1.success and result_2.success
    model_1 = build_projector(result_1)
    model_2 = build_projector(result_2)
    for u, v in ((640.0, 400.0), (300.0, 200.0), (900.0, 600.0)):
        assert model_1.unproject_pixel(u, v) == pytest.approx(model_2.unproject_pixel(u, v), abs=1e-5)
    assert result_1.fitted_params["neural_best_epoch"] == pytest.approx(result_2.fitted_params["neural_best_epoch"])


# ---------------------------------------------------------------------------
# Test J - Different Seed: finite한 모델 + Seed Stability 계산 가능
# ---------------------------------------------------------------------------

def test_different_seeds_produce_finite_models_and_seed_stability_is_computable():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]

    mean_deg, p95_deg = compute_seed_stability_deg(
        dataset, _neural_config(K, D), camera_config, train_ids, test_ids, seeds=(1, 2),
    )

    assert mean_deg is not None and np.isfinite(mean_deg) and mean_deg >= 0
    assert p95_deg is not None and np.isfinite(p95_deg) and p95_deg >= mean_deg - 1e-9


def test_seed_stability_is_zero_for_identical_models():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]
    result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)
    assert result.success
    model = build_projector(result)

    # float32(torch) 왕복 변환에서 오는 미세한 부동소수점 노이즈가 있을 수
    # 있어(동일 모델을 두 번 평가해도 완전히 0은 아닐 수 있다) 느슨한
    # 허용오차를 쓴다 - Grid/RBF(float64, 순수 결정론적 보간)보다 관대한
    # tolerance가 필요하다.
    mean_deg, p95_deg = compute_ray_stability_deg([model, model], 1280.0, 800.0)
    assert mean_deg == pytest.approx(0.0, abs=1e-4)
    assert p95_deg == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test K - STAGE B pose diagnostics finite/bounded
# ---------------------------------------------------------------------------

def test_neural_stage_b_pose_diagnostics_are_finite_and_bounded():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)

    assert result.success, result.error_message
    for key in (
        "diag_pose_delta_r_median_deg", "diag_pose_delta_r_p95_deg",
        "diag_pose_delta_t_median_mm", "diag_pose_delta_t_p95_mm",
    ):
        assert key in result.fitted_params
        assert np.isfinite(result.fitted_params[key])
    assert result.fitted_params["diag_pose_delta_r_p95_deg"] < 30.0
    assert result.fitted_params["diag_pose_delta_t_p95_mm"] < 500.0
    if result.fitted_params["stage_used_is_joint_refined"] == 0.0:
        assert result.fitted_params["diag_pose_delta_r_median_deg"] == pytest.approx(0.0, abs=1e-9)
        assert result.fitted_params["diag_pose_delta_t_median_mm"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test M/N - Repeated Hold-out + Ray Stability
# ---------------------------------------------------------------------------

def test_neural_repeated_holdout_runs_and_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()

    summary = run_repeated_holdout_neural_residual(
        dataset, _neural_config(K, D), camera_config, seeds=(1, 2, 3), test_ratio=0.3,
    )

    assert summary.n_successful >= 2
    assert summary.mean_test_rmse is not None and np.isfinite(summary.mean_test_rmse)
    assert summary.std_test_rmse is not None and summary.std_test_rmse >= 0
    if summary.mean_test_p95 is not None:
        assert np.isfinite(summary.mean_test_p95)
    assert summary.ray_stability_mean_deg is not None
    assert summary.ray_stability_p95_deg is not None
    assert np.isfinite(summary.ray_stability_mean_deg)
    assert summary.ray_stability_p95_deg >= summary.ray_stability_mean_deg - 1e-9
    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Test O/P - Grid/RBF/Neural이 같은 outer split + 같은 synthetic GT를 쓰는지
# ---------------------------------------------------------------------------

def test_grid_rbf_neural_use_the_same_outer_split_for_fair_comparison():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=4)

    grid_result = calibrate_residual_ray(dataset, _config(K, D), camera_config, train_ids, test_ids)
    rbf_result = calibrate_residual_rbf(
        dataset, _config(K, D, residual_ray_hint={"method": "rbf", "rbf_num_centers": 8.0, "rbf_smoothing": 1e-4}),
        camera_config, train_ids, test_ids,
    )
    neural_result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)

    assert grid_result.success and rbf_result.success and neural_result.success
    assert grid_result.train_frame_ids == rbf_result.train_frame_ids == neural_result.train_frame_ids == train_ids
    assert grid_result.test_frame_ids == rbf_result.test_frame_ids == neural_result.test_frame_ids == test_ids


def test_same_synthetic_gt_grid_rbf_neural_all_produce_valid_runtime_models():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=2)

    grid_result = calibrate_residual_ray(dataset, _config(K, D), camera_config, train_ids, test_ids)
    rbf_result = calibrate_residual_rbf(
        dataset, _config(K, D, residual_ray_hint={"method": "rbf", "rbf_num_centers": 8.0, "rbf_smoothing": 1e-4}),
        camera_config, train_ids, test_ids,
    )
    neural_result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)

    assert grid_result.success and rbf_result.success and neural_result.success
    for result in (grid_result, rbf_result, neural_result):
        assert result.test_residual_stats and np.isfinite(result.test_residual_stats.rmse)

    for model in (build_projector(grid_result), build_projector(rbf_result), build_projector(neural_result)):
        ray = model.unproject_pixel(700.0, 380.0)
        uv = model.project_point(0.15, -0.1, 5.0)
        assert np.all(np.isfinite(ray))
        assert np.all(np.isfinite(uv))

    # 이 테스트는 세 모델이 반드시 서로 우열을 가려야 한다고 주장하지
    # 않는다(사용자 스펙 49번, "Neural이 반드시 더 좋다는 assertion 금지") -
    # 셋 다 유효한 runtime 모델을 만든다는 것만 확인한다.


# ---------------------------------------------------------------------------
# Repeated Hold-out / Seed Stability = Outer Train subset만 (진단
# 오케스트레이터의 leakage 회귀 테스트)
# ---------------------------------------------------------------------------

def test_neural_diagnostics_repeated_holdout_uses_outer_train_subset_only(monkeypatch):
    import calibration.windshield.neural_residual as neural_module

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    seen_ids: list[str] = []

    def fake_repeated(inner_dataset, config, camera_config_arg, seeds, test_ratio):
        seen_ids.extend(f.image_info.image_id for f in inner_dataset.frames)
        from calibration.windshield.residual_common import RepeatedHoldoutSummary

        return RepeatedHoldoutSummary(seeds_used=list(seeds), n_successful=0)

    monkeypatch.setattr(neural_module, "run_repeated_holdout_neural_residual", fake_repeated)

    result = run_neural_residual_calibration_with_diagnostics(
        dataset, _neural_config(K, D), camera_config, train_ids, test_ids,
        compute_repeated_holdout=True, repeated_holdout_seeds=(1,),
        compute_seed_stability=False,
    )

    assert result.success, result.error_message
    assert seen_ids
    assert set(seen_ids) == set(train_ids)
    assert not (set(seen_ids) & set(test_ids))


def test_neural_diagnostics_seed_stability_never_receives_outer_test_ids(monkeypatch):
    """compute_seed_stability_deg는 train_ids/test_ids를 그대로 받아 Test는
    항상 pose-only 평가에만 쓰인다(leakage 없음) - 실제로 넘어오는 인자를
    monkeypatch로 캡처해 test_ids가 학습(backprop)에 쓰이는 calibrate_neural_
    residual 호출 자체의 train_ids와 절대 섞이지 않는지 확인한다."""
    import calibration.windshield.neural_residual as neural_module

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    captured_train_ids: list[list[str]] = []

    original_calibrate = neural_module.calibrate_neural_residual

    def spying_calibrate(windshield_dataset, config, camera_config_arg, train_ids_arg, test_ids_arg):
        captured_train_ids.append(list(train_ids_arg))
        assert not (set(train_ids_arg) & set(test_ids))
        return original_calibrate(windshield_dataset, config, camera_config_arg, train_ids_arg, test_ids_arg)

    monkeypatch.setattr(neural_module, "calibrate_neural_residual", spying_calibrate)

    mean_deg, _p95 = neural_module.compute_seed_stability_deg(
        dataset, _neural_config(K, D), camera_config, train_ids, test_ids, seeds=(1, 2),
    )

    assert len(captured_train_ids) == 2
    for captured in captured_train_ids:
        assert set(captured) == set(train_ids)


# ---------------------------------------------------------------------------
# neural_batch_size 재현성 (STEP 5 안정화 라운드 항목 3)
# ---------------------------------------------------------------------------

def test_neural_batch_size_is_persisted_in_fitted_params():
    """항목 3-1 - 실제 학습에 쓰인 batch_size가 fitted_params에 저장돼야
    재현/재구성 시 기본값(128)으로 조용히 되돌아가지 않는다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]

    result = calibrate_neural_residual(
        dataset, _neural_config(K, D, {"neural_batch_size": 32}), camera_config, train_ids, test_ids,
    )

    assert result.success, result.error_message
    assert result.fitted_params["neural_batch_size"] == 32.0


def test_neural_batch_size_is_forwarded_to_repeated_holdout_resolved_config(monkeypatch):
    """항목 3-2 - Main fit이 batch_size=32로 학습했다면, Repeated Hold-out에
    실제로 넘어가는 resolved_config도 32여야 한다(기본 128로 되돌아가면
    안 된다). run_repeated_holdout_neural_residual을 monkeypatch해서 실제로
    받는 config.residual_ray_hint를 직접 캡처한다."""
    import calibration.windshield.neural_residual as neural_module

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    captured_hints: list[dict] = []

    def fake_repeated(inner_dataset, config, camera_config_arg, seeds, test_ratio):
        captured_hints.append(dict(config.residual_ray_hint))
        from calibration.windshield.residual_common import RepeatedHoldoutSummary

        return RepeatedHoldoutSummary(seeds_used=list(seeds), n_successful=0)

    monkeypatch.setattr(neural_module, "run_repeated_holdout_neural_residual", fake_repeated)

    result = run_neural_residual_calibration_with_diagnostics(
        dataset, _neural_config(K, D, {"neural_batch_size": 32}), camera_config, train_ids, test_ids,
        compute_repeated_holdout=True, repeated_holdout_seeds=(1,),
        compute_seed_stability=False,
    )

    assert result.success, result.error_message
    assert result.fitted_params["neural_batch_size"] == 32.0
    assert len(captured_hints) == 1
    assert captured_hints[0]["neural_batch_size"] == 32.0


def test_neural_batch_size_smaller_than_dataset_still_trains_correctly():
    """작은 batch_size(<n_train)로도 mini-batch 루프가 정상 동작하는지 -
    `eff_batch = min(batch_size, n_train)` 정책이 실제로 mini-batch를 여러
    번 도는 경로를 taken하는지 확인(batch_size=8이면 대부분의 합성
    fixture에서 코너 수보다 훨씬 작다)."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]

    result = calibrate_neural_residual(
        dataset, _neural_config(K, D, {"neural_batch_size": 8}), camera_config, train_ids, test_ids,
    )

    assert result.success, result.error_message
    assert result.fitted_params["neural_batch_size"] == 8.0
    assert np.isfinite(result.fitted_params["neural_best_train_ray_loss"])
