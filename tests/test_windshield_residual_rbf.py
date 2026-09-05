from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.projection import build_projector
from calibration.windshield.residual_common import RepeatedHoldoutSummary, normalize_pixel_coordinates
from calibration.windshield.residual_rbf import (
    ResidualRBFWindshieldModel,
    calibrate_residual_rbf,
    select_best_rbf_hyperparams,
)
from calibration.windshield.validation import run_windshield_calibration
from export.windshield import export_windshield_yaml, windshield_model_from_yaml
from tests._windshield_test_utils import (
    build_synthetic_residual_ray_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
    default_residual_delta_fn,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _square_centers() -> np.ndarray:
    return np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float64)


def _config(K, D, **kwargs) -> WindshieldConfig:
    return WindshieldConfig(
        base_model_name=_MODEL,
        base_camera_matrix=K,
        base_distortion=D,
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        **kwargs,
    )


def test_normalized_pixel_coordinates_uses_minus_one_plus_one_convention():
    assert normalize_pixel_coordinates(0.0, 0.0, 1280.0, 800.0) == pytest.approx((-1.0, -1.0))
    assert normalize_pixel_coordinates(640.0, 400.0, 1280.0, 800.0) == pytest.approx((0.0, 0.0))
    assert normalize_pixel_coordinates(1280.0, 800.0, 1280.0, 800.0) == pytest.approx((1.0, 1.0))


def test_zero_residual_rbf_reproduces_baseline():
    K, D = default_camera_matrix_distortion()
    model = ResidualRBFWindshieldModel(
        K, D, _MODEL, _square_centers(), np.zeros((4, 3)), 1280, 800, smoothing=0.0
    )
    baseline = BaselineWindshieldModel(K, D, _MODEL)

    assert model.unproject_pixel(700.0, 380.0) == pytest.approx(
        baseline.unproject_pixel(700.0, 380.0), abs=1e-8
    )
    assert model.project_point(0.2, 0.15, 5.0) == pytest.approx(
        baseline.project_point(0.2, 0.15, 5.0), abs=1e-6
    )


def test_known_rbf_interpolation_is_finite_and_matches_centers_when_unsmoothed():
    K, D = default_camera_matrix_distortion()
    centers = _square_centers()
    values = np.array(
        [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03], [0.01, 0.02, 0.03]],
        dtype=np.float64,
    )
    model = ResidualRBFWindshieldModel(K, D, _MODEL, centers, values, 1280, 800, smoothing=0.0)

    for u, v in ((0.0, 0.0), (1280.0, 0.0), (0.0, 800.0), (1280.0, 800.0), (640.0, 400.0)):
        delta = model._delta(u, v)  # noqa: SLF001
        assert np.all(np.isfinite(delta))

    assert model._delta(0.0, 0.0) == pytest.approx(values[0], abs=1e-8)  # noqa: SLF001
    assert model._delta(1280.0, 800.0) == pytest.approx(values[3], abs=1e-8)  # noqa: SLF001


def test_build_projector_and_yaml_round_trip_reconstruct_rbf(tmp_path):
    K, D = default_camera_matrix_distortion()
    fitted_params = {
        "residual_ray_method": 1.0,
        "image_width": 1280.0,
        "image_height": 800.0,
        "rbf_kernel_code": 0.0,
        "rbf_smoothing": 0.0,
        "rbf_num_centers": 4.0,
        "runtime_param_count": 12.0,
    }
    for i, (center, value) in enumerate(zip(_square_centers(), np.zeros((4, 3)))):
        fitted_params[f"rbf_center_u_{i}"] = float(center[0])
        fitted_params[f"rbf_center_v_{i}"] = float(center[1])
        fitted_params[f"rbf_residual_dx_{i}"] = float(value[0])
        fitted_params[f"rbf_residual_dy_{i}"] = float(value[1])
        fitted_params[f"rbf_residual_dz_{i}"] = float(value[2])

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        base_model_name=_MODEL,
        base_camera_matrix=K,
        base_distortion=D,
        fitted_params=fitted_params,
        success=True,
    )
    original = build_projector(result)
    path = str(tmp_path / "windshield_rbf.yml")
    export_windshield_yaml(result, default_camera_config(), path)
    loaded = windshield_model_from_yaml(path)

    assert loaded.unproject_pixel(700.0, 380.0) == pytest.approx(
        original.unproject_pixel(700.0, 380.0), abs=1e-8
    )
    assert loaded.project_point(0.2, -0.1, 5.0) == pytest.approx(
        original.project_point(0.2, -0.1, 5.0), abs=1e-6
    )


def test_run_windshield_calibration_dispatches_rbf_method(monkeypatch):
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    called = {"rbf": False}

    def fake_calibrate(dataset_arg, config_arg, camera_config_arg, train_ids, test_ids):
        called["rbf"] = True
        return WindshieldCalibrationResult(
            windshield_model=WindshieldModelType.RESIDUAL_RAY,
            base_model_name=config_arg.base_model_name,
            base_camera_matrix=config_arg.base_camera_matrix,
            base_distortion=config_arg.base_distortion,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            fitted_params={"residual_ray_method": 1.0},
            success=True,
        )

    import calibration.windshield.residual_rbf as residual_rbf_module

    monkeypatch.setattr(residual_rbf_module, "calibrate_residual_rbf", fake_calibrate)
    result = run_windshield_calibration(
        dataset,
        _config(K, D, residual_ray_hint={"method": "rbf", "rbf_num_centers": 4.0}),
        camera_config,
    )

    assert called["rbf"] is True
    assert result.success
    assert result.fitted_params["residual_ray_method"] == 1.0


def test_rbf_auto_selection_prefers_fewer_centers_within_tolerance(monkeypatch):
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]

    def fake_holdout(dataset_arg, config_arg, camera_config_arg, seeds, test_ratio):
        centers = int(config_arg.residual_ray_hint["rbf_num_centers"])
        rmse = 1.0 if centers == 8 else 0.98
        return RepeatedHoldoutSummary(seeds_used=list(seeds), n_successful=len(seeds), mean_test_rmse=rmse)

    import calibration.windshield.residual_rbf as residual_rbf_module

    monkeypatch.setattr(residual_rbf_module, "run_repeated_holdout_residual_rbf", fake_holdout)
    (centers, smoothing), candidates = select_best_rbf_hyperparams(
        dataset,
        _config(K, D),
        camera_config,
        train_ids,
        center_candidates=[8, 16],
        smoothing_candidates=[1e-4],
        seeds=(1,),
    )

    assert (centers, smoothing) == (8, 1e-4)
    assert [c.param_count for c in candidates] == [24, 48]


def test_calibrate_residual_rbf_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]

    result = calibrate_residual_rbf(
        dataset,
        _config(K, D, residual_ray_hint={"method": "rbf", "rbf_num_centers": 8.0, "rbf_smoothing": 1e-4}),
        camera_config,
        train_ids,
        test_ids,
    )

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)
    assert result.success, result.error_message
    assert result.fitted_params["residual_ray_method"] == 1.0
    assert result.fitted_params["rbf_num_centers"] >= 3.0
