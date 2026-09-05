"""
tests/test_windshield_base.py
==================================

calibration.windshield.base의 타입/dispatch 스캐폴딩 검증.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.windshield.base import (
    WindshieldCalibrationResult,
    WindshieldConfig,
    WindshieldModelType,
)
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.projection import build_projector
from calibration.windshield.spherical import SphericalWindshieldModel


def _make_config() -> WindshieldConfig:
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    return WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)


def test_windshield_config_defaults():
    cfg = _make_config()
    assert cfg.windshield_model == WindshieldModelType.BASELINE
    assert cfg.test_ratio == 0.25
    assert cfg.split_seed == 42
    # Phase 2+ 전용 필드는 아직 아무도 쓰지 않지만 기본값은 None이어야 한다.
    assert cfg.glass_refractive_index is None
    assert cfg.glass_thickness_m is None
    assert cfg.windshield_position_hint is None


def test_windshield_calibration_result_defaults():
    cfg = _make_config()
    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.BASELINE,
        base_model_name=cfg.base_model_name,
        base_camera_matrix=cfg.base_camera_matrix,
        base_distortion=cfg.base_distortion,
    )
    assert result.fitted_params == {}
    assert result.success is False
    assert result.train_frame_ids == []
    assert result.test_frame_ids == []
    assert result.residual_stats is None


def test_build_projector_baseline_returns_baseline_model():
    cfg = _make_config()
    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.BASELINE,
        base_model_name=cfg.base_model_name,
        base_camera_matrix=cfg.base_camera_matrix,
        base_distortion=cfg.base_distortion,
        success=True,
    )
    model = build_projector(result)
    assert isinstance(model, BaselineWindshieldModel)


def test_build_projector_spherical_returns_spherical_model():
    """Phase 2(Spherical)는 STEP 2에서 실제 구현됐다 - fitted_params가 채워진
    결과라면 build_projector가 실제 SphericalWindshieldModel을 만들어야 한다."""
    cfg = _make_config()
    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPHERICAL,
        base_model_name=cfg.base_model_name,
        base_camera_matrix=cfg.base_camera_matrix,
        base_distortion=cfg.base_distortion,
        fitted_params={
            "sphere_center_x": 0.0, "sphere_center_y": 0.0, "sphere_center_z": -9.7,
            "sphere_radius": 10.0,
        },
        success=True,
    )
    model = build_projector(result)
    assert isinstance(model, SphericalWindshieldModel)


@pytest.mark.parametrize(
    "windshield_model",
    [WindshieldModelType.RESIDUAL_RAY, WindshieldModelType.SPLINE],
)
def test_build_projector_unimplemented_models_raise_not_implemented(windshield_model):
    cfg = _make_config()
    result = WindshieldCalibrationResult(
        windshield_model=windshield_model,
        base_model_name=cfg.base_model_name,
        base_camera_matrix=cfg.base_camera_matrix,
        base_distortion=cfg.base_distortion,
        success=True,
    )
    with pytest.raises(NotImplementedError):
        build_projector(result)
