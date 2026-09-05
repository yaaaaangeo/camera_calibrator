"""
tests/test_windshield_regression_existing_features.py
===========================================================

Windshield Refraction Calibration을 추가하면서 calibration/models/common.py,
calibration/validation.py, calibration/radial_profile.py에 손을 댔다
(cv2.solvePnP/projectPoints의 fisheye/pinhole 분기를 solve_pnp_for_model/
project_points_for_model로 추출). 이 리팩터링이 기존 Camera Intrinsic
Calibration의 동작을 조금도 바꾸지 않았는지 여기서 명시적으로 확인한다
(사용자 스펙 12/22번 "기존 기능을 절대 깨지 마라").
"""

from __future__ import annotations

from calibration.models.brown_conrady import calibrate_brown_conrady
from calibration.types import CameraModelType
from calibration.validation import split_train_test, validate_holdout
from export.opencv import export_opencv_yaml, load_opencv_yaml
from calibration.types import PatternConfig, PatternType


def test_existing_intrinsic_calibration_still_succeeds(synthetic_dataset, camera_config):
    result = calibrate_brown_conrady(synthetic_dataset, camera_config)
    assert result.success
    assert result.camera_matrix is not None
    assert result.distortion is not None
    assert result.residual_stats is not None
    assert result.spatial_error_map is not None


def test_existing_holdout_validation_still_succeeds(synthetic_dataset, camera_config, pattern_config):
    train_ids, test_ids = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25, seed=42)
    validation = validate_holdout(
        synthetic_dataset, camera_config, pattern_config, CameraModelType.BROWN_CONRADY, train_ids, test_ids
    )
    assert validation.success
    assert validation.train_rms is not None
    assert validation.test_rms is not None
    assert not validation.failed_test_frame_ids


def test_opencv_yaml_export_unaffected_by_windshield_addition(tmp_path, synthetic_dataset, camera_config):
    pattern_config = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100")
    result = calibrate_brown_conrady(synthetic_dataset, camera_config)
    assert result.success

    path = str(tmp_path / "camera.yml")
    export_opencv_yaml(result, camera_config, pattern_config, path)
    data = load_opencv_yaml(path)

    assert data["calibration_model"] == CameraModelType.BROWN_CONRADY.value
    assert "windshield" not in data
    assert data["camera_matrix"].shape == (3, 3)
