"""
tests/test_windshield_export.py
====================================

export/windshield.py::export_windshield_yaml 검증. export/opencv.py의 기존
OpenCV YAML(camera_matrix/distortion_coefficients)과는 완전히 별개의 파일/
스키마여야 한다(사용자 스펙 19번) - 이 테스트는 새 스키마의 round-trip만 본다.
export/opencv.py 자체가 영향을 받지 않는지는
test_windshield_regression_existing_features.py에서 확인한다.
"""

from __future__ import annotations

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import calibrate_baseline
from export.windshield import export_windshield_yaml, load_windshield_yaml
from tests._windshield_test_utils import (
    build_synthetic_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
)


def _baseline_result():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)
    train_ids = [f.image_info.image_id for f in dataset.frames]
    result = calibrate_baseline(dataset, config, camera_config, train_ids, [])
    return result, camera_config


def test_export_windshield_yaml_round_trip(tmp_path):
    result, camera_config = _baseline_result()
    path = str(tmp_path / "windshield.yml")

    export_windshield_yaml(result, camera_config, path)
    data = load_windshield_yaml(path)

    assert data["base_camera"]["camera_model"] == CameraModelType.BROWN_CONRADY.value
    assert data["base_camera"]["camera_matrix"].shape == (3, 3)
    assert data["base_camera"]["image_width"] == camera_config.width
    assert data["windshield"]["model"] == WindshieldModelType.BASELINE.value
    assert data["windshield"]["train_rms"] >= 0.0


def test_export_windshield_yaml_rejects_failed_result(tmp_path):
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)
    failed_result = calibrate_baseline(dataset, config, camera_config, [], [])
    assert not failed_result.success

    try:
        export_windshield_yaml(failed_result, camera_config, str(tmp_path / "should_not_exist.yml"))
        assert False, "실패한 결과는 export가 거부해야 합니다."
    except ValueError:
        pass
