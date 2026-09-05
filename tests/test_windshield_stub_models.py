"""
tests/test_windshield_stub_models.py
=========================================

Phase 4(Spline)는 이번 라운드에 구현하지 않는다 - 대신 스캐폴딩(enum
dispatch, 클래스/함수 시그니처)이 "구현되지 않았다"는 걸 명확한 예외로
알리는지만 확인한다. Phase 2(Spherical)는 STEP 2에서, Phase 3-A(Residual
Ray Grid)는 이번 라운드에서 실제 구현됐으므로 이 파일에서 제외됐다 - 각각
tests/test_windshield_spherical.py, tests/test_windshield_residual_ray.py를
본다.
"""

from __future__ import annotations

import pytest

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldConfig
from calibration.windshield.spline import SplineWindshieldModel, calibrate_spline
from tests._windshield_test_utils import build_synthetic_windshield_dataset, default_camera_config, default_camera_matrix_distortion


def test_stub_model_constructor_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        SplineWindshieldModel()


def test_stub_calibrate_function_raises_not_implemented():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    with pytest.raises(NotImplementedError):
        calibrate_spline(dataset, config, camera_config, train_ids, [])


def test_run_windshield_calibration_dispatches_unimplemented_models_cleanly():
    from calibration.windshield.base import WindshieldModelType
    from calibration.windshield.validation import run_windshield_calibration

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY,
        base_camera_matrix=K,
        base_distortion=D,
        windshield_model=WindshieldModelType.SPLINE,
    )

    with pytest.raises(NotImplementedError):
        run_windshield_calibration(dataset, config, camera_config)
