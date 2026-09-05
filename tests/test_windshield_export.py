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

import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import calibrate_baseline
from calibration.windshield.projection import build_projector
from calibration.windshield.spherical import calibrate_spherical
from calibration.windshield.spline import calibrate_spline
from export.windshield import export_windshield_yaml, load_windshield_yaml, windshield_model_from_yaml
from tests._windshield_test_utils import (
    build_synthetic_spherical_windshield_dataset,
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


def test_export_windshield_yaml_round_trip_spherical(tmp_path):
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPHERICAL,
        windshield_position_hint={"sphere_center_z": -8.0, "sphere_radius": 9.0},
    )
    train_ids = [f.image_info.image_id for f in dataset.frames]
    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])
    assert result.success

    path = str(tmp_path / "windshield_spherical.yml")
    export_windshield_yaml(result, camera_config, path)
    data = load_windshield_yaml(path)

    assert data["windshield"]["model"] == WindshieldModelType.SPHERICAL.value
    assert data["windshield"]["fitted_params"]["sphere_radius"] == pytest.approx(
        result.fitted_params["sphere_radius"]
    )

    reconstructed = windshield_model_from_yaml(path)
    test_point = (0.2, 0.15, 5.0)
    original_model = build_projector(result)
    assert reconstructed.project_point(*test_point) == pytest.approx(
        original_model.project_point(*test_point), abs=1e-6
    )


def test_export_windshield_yaml_round_trip_spline(tmp_path):
    """STEP 4(Spline) - Test N: export -> load -> runtime reconstruction 후
    project_point/unproject_pixel이 원본과 tolerance 내에서 일치해야 한다.
    SciPy 내부 상태가 아니라 base sphere + spline grid 값(재구성 가능한
    public 값)만 저장한다는 것도 이 round-trip이 성공한다는 사실로 검증된다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPLINE, spline_hint={"spline_rows": 2.0, "spline_cols": 2.0},
    )
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)
    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)
    assert result.success, result.error_message

    path = str(tmp_path / "windshield_spline.yml")
    export_windshield_yaml(result, camera_config, path)
    data = load_windshield_yaml(path)

    assert data["windshield"]["model"] == WindshieldModelType.SPLINE.value
    assert data["windshield"]["fitted_params"]["sphere_radius"] == pytest.approx(
        result.fitted_params["sphere_radius"]
    )

    reconstructed = windshield_model_from_yaml(path)
    original_model = build_projector(result)
    for test_point in [(0.05, 0.02, 3.0), (-0.1, 0.05, 4.0), (0.0, -0.05, 2.5)]:
        assert reconstructed.project_point(*test_point) == pytest.approx(
            original_model.project_point(*test_point), abs=1e-4
        )
    for u, v in [(640.0, 400.0), (300.0, 200.0), (900.0, 600.0)]:
        assert reconstructed.unproject_pixel(u, v) == pytest.approx(
            original_model.unproject_pixel(u, v), abs=1e-6
        )


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
