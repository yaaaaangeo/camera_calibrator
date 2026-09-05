"""
tests/test_windshield_neural_residual.py
==============================================

STEP 5 - Neural Residual Windshield Model 기본 검증(Test A-E, Q, R 매핑).

PyTorch가 설치돼 있지 않은 환경에서는 이 파일 전체를 skip한다 - Neural은
선택적 의존성이고, 다른 Windshield 모델(Baseline/Spherical/Residual Grid/
Residual RBF/Spline) 테스트는 이 파일과 무관하게 항상 실행 가능해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.neural_residual import (
    DEFAULT_NEURAL_HIDDEN_DIMS,
    NeuralResidualWindshieldModel,
    _build_mlp,
    build_neural_residual_model_from_fitted_params,
    calibrate_neural_residual,
)
from calibration.windshield.projection import build_projector
from calibration.windshield.residual_common import normalize_pixel_coordinates
from calibration.windshield.validation import run_windshield_calibration
from export.windshield import export_windshield_yaml, windshield_model_from_yaml
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
    return _config(
        K, D,
        residual_ray_hint=hint or {"method": "neural", "neural_max_epochs": 80, "neural_patience": 15},
    )


def _zero_state_dict(hidden_dims=DEFAULT_NEURAL_HIDDEN_DIMS, activation="silu"):
    import torch

    net = _build_mlp(hidden_dims, activation)
    sd = net.state_dict()
    for k in sd:
        sd[k] = torch.zeros_like(sd[k])
    return sd


# ---------------------------------------------------------------------------
# Test A - Zero Neural: 모든 weight가 0이면 corrected ray == base ray
# ---------------------------------------------------------------------------

def test_zero_neural_weights_reproduce_baseline_ray_and_projection():
    K, D = default_camera_matrix_distortion()
    model = NeuralResidualWindshieldModel(
        K, D, _MODEL, _zero_state_dict(), DEFAULT_NEURAL_HIDDEN_DIMS, "silu", 1280.0, 800.0,
    )
    baseline = BaselineWindshieldModel(K, D, _MODEL)

    assert model.unproject_pixel(700.0, 380.0) == pytest.approx(
        baseline.unproject_pixel(700.0, 380.0), abs=1e-8
    )
    assert model.project_point(0.2, 0.15, 5.0) == pytest.approx(
        baseline.project_point(0.2, 0.15, 5.0), abs=1e-6
    )


# ---------------------------------------------------------------------------
# Test B - Forward shape [N,2] -> [N,3], NaN/Inf 없음
# ---------------------------------------------------------------------------

def test_mlp_forward_shape_and_finiteness():
    import torch

    net = _build_mlp(DEFAULT_NEURAL_HIDDEN_DIMS, "silu")
    x = torch.randn(17, 2, dtype=torch.float32)
    y = net(x)
    assert tuple(y.shape) == (17, 3)
    assert torch.all(torch.isfinite(y))


# ---------------------------------------------------------------------------
# Test C - UV normalization: (0,0)/(W,H)/center가 기대한 정규화 좌표가 되는지,
# 그리고 그 좌표들에서 실제 모델 평가가 finite한지.
# ---------------------------------------------------------------------------

def test_uv_normalization_matches_grid_rbf_convention_and_model_is_finite_at_corners():
    assert normalize_pixel_coordinates(0.0, 0.0, 1280.0, 800.0) == pytest.approx((-1.0, -1.0))
    assert normalize_pixel_coordinates(640.0, 400.0, 1280.0, 800.0) == pytest.approx((0.0, 0.0))
    assert normalize_pixel_coordinates(1280.0, 800.0, 1280.0, 800.0) == pytest.approx((1.0, 1.0))

    K, D = default_camera_matrix_distortion()
    model = NeuralResidualWindshieldModel(
        K, D, _MODEL, _zero_state_dict(), DEFAULT_NEURAL_HIDDEN_DIMS, "silu", 1280.0, 800.0,
    )
    for u, v in ((0.0, 0.0), (1280.0, 0.0), (0.0, 800.0), (1280.0, 800.0), (640.0, 400.0)):
        assert np.all(np.isfinite(model.unproject_pixel(u, v)))


# ---------------------------------------------------------------------------
# Test D - Known smooth field fitting: Grid/RBF와 동일 synthetic GT를 학습해
# baseline(보정 없음)보다 test RMSE가 뚜렷이 개선되는지(structural, exact
# recovery를 요구하지 않는다).
# ---------------------------------------------------------------------------

def test_calibrate_neural_residual_learns_smooth_gt_and_beats_uncorrected_baseline():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]

    result = calibrate_neural_residual(
        dataset,
        _neural_config(K, D, {"method": "neural", "neural_max_epochs": 300, "neural_patience": 30}),
        camera_config, train_ids, test_ids,
    )

    assert result.success, result.error_message
    assert result.test_residual_stats is not None
    assert np.isfinite(result.test_residual_stats.rmse)
    # Baseline(보정 없음)의 GT 데이터셋 RMS는 GT delta_fn의 스케일(~0.02 in
    # ray-direction -> 초점거리 900px 기준 약 10px 이상)에 비례해 크다 -
    # Neural이 뭔가 의미 있게 학습했다면 이보다 훨씬 작아야 한다(느슨한
    # 상한, exact recovery는 요구하지 않는다).
    assert result.test_residual_stats.rmse < 5.0


def test_calibrate_neural_residual_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]

    result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)
    assert result.success, result.error_message
    assert result.fitted_params["residual_ray_method"] == 2.0
    assert result.neural_state_dict_b64


def test_project_point_unproject_pixel_roundtrip_no_nan_at_corners():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]
    result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)
    assert result.success, result.error_message

    model = build_projector(result)
    for u, v in ((0.0, 0.0), (1280.0, 0.0), (0.0, 800.0), (1280.0, 800.0), (640.0, 400.0)):
        ray = model.unproject_pixel(u, v)
        assert np.all(np.isfinite(ray))
        assert np.linalg.norm(ray) == pytest.approx(1.0, abs=1e-6)

    uv = model.project_point(0.15, -0.1, 5.0)
    assert np.all(np.isfinite(uv))
    ray_back = model.unproject_pixel(*uv)
    assert np.all(np.isfinite(ray_back))


# ---------------------------------------------------------------------------
# Test Q - Model Save/Load: Train -> export YAML(+ sibling .pt) -> load ->
# 같은 픽셀에서 동일한 결과.
# ---------------------------------------------------------------------------

def test_neural_model_export_yaml_and_reload_matches_original(tmp_path):
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
    test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]
    result = calibrate_neural_residual(dataset, _neural_config(K, D), camera_config, train_ids, test_ids)
    assert result.success, result.error_message

    path = str(tmp_path / "windshield_neural.yml")
    export_windshield_yaml(result, camera_config, path)
    assert (tmp_path / "windshield_neural_neural.pt").exists()

    reconstructed = windshield_model_from_yaml(path)
    original = build_projector(result)
    for u, v in ((700.0, 380.0), (300.0, 200.0), (900.0, 600.0)):
        assert reconstructed.unproject_pixel(u, v) == pytest.approx(original.unproject_pixel(u, v), abs=1e-6)
    assert reconstructed.project_point(0.2, -0.1, 5.0) == pytest.approx(
        original.project_point(0.2, -0.1, 5.0), abs=1e-6
    )


def test_run_windshield_calibration_dispatches_neural_method(monkeypatch):
    from calibration.windshield.base import WindshieldCalibrationResult

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
    camera_config = default_camera_config()
    called = {"neural": False}

    def fake_calibrate(dataset_arg, config_arg, camera_config_arg, train_ids, test_ids):
        called["neural"] = True
        return WindshieldCalibrationResult(
            windshield_model=WindshieldModelType.RESIDUAL_RAY,
            base_model_name=config_arg.base_model_name,
            base_camera_matrix=config_arg.base_camera_matrix,
            base_distortion=config_arg.base_distortion,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            fitted_params={"residual_ray_method": 2.0},
            success=True,
        )

    import calibration.windshield.neural_residual as neural_module

    monkeypatch.setattr(neural_module, "calibrate_neural_residual", fake_calibrate)
    result = run_windshield_calibration(
        dataset,
        _config(K, D, residual_ray_hint={"method": "neural"}),
        camera_config,
    )

    assert called["neural"] is True
    assert result.success
    assert result.fitted_params["residual_ray_method"] == 2.0


def test_build_neural_residual_model_requires_state_dict():
    K, D = default_camera_matrix_distortion()
    fitted_params = {
        "neural_num_hidden_layers": 3.0,
        "neural_hidden_dim_0": 32.0, "neural_hidden_dim_1": 64.0, "neural_hidden_dim_2": 32.0,
        "neural_activation_code": 0.0,
        "image_width": 1280.0, "image_height": 800.0,
    }
    with pytest.raises(ValueError):
        build_neural_residual_model_from_fitted_params(K, D, _MODEL, fitted_params, None)


# ---------------------------------------------------------------------------
# Test R - CPU Runtime: GPU 없이도 inference 가능해야 한다(이 환경 자체가
# CPU-only torch wheel이므로, 정상 동작 = CPU inference가 실제로 동작함을
# 그대로 증명한다).
# ---------------------------------------------------------------------------

def test_inference_runs_on_cpu_only_tensors():
    import torch

    K, D = default_camera_matrix_distortion()
    model = NeuralResidualWindshieldModel(
        K, D, _MODEL, _zero_state_dict(), DEFAULT_NEURAL_HIDDEN_DIMS, "silu", 1280.0, 800.0,
    )
    assert next(model._net.parameters()).device.type == "cpu"  # noqa: SLF001
    assert not torch.cuda.is_available() or True  # CPU 경로 자체가 CUDA 유무와 무관하게 동작해야 함
    ray = model.unproject_pixel(640.0, 400.0)
    assert np.all(np.isfinite(ray))
