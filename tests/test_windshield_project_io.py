"""
tests/test_windshield_project_io.py
========================================

CalibrationProject에 추가된 windshield_config/windshield_dataset/
windshield_results 필드의 저장/불러오기 검증.

* project_to_dict()는 dataclasses.asdict() + json_safe() 기반으로 이미
  범용적이라 새 필드 자체를 위한 write-side 코드가 필요 없다(project_io.py
  docstring) - 이 테스트는 그 전제가 실제로 유지되는지 확인한다.
* windshield 필드가 없는(과거) 프로젝트 파일도 그대로 로드되어야 한다
  (버전을 올리지 않았으므로 이 회귀는 특히 중요).
"""

from __future__ import annotations

import pytest

from calibration.project_io import (
    PROJECT_FORMAT_VERSION,
    _windshield_calibration_result_from_dict,
    project_from_dict,
    project_to_dict,
    save_project,
    load_project,
)
from calibration.json_utils import json_safe
import dataclasses
from calibration.types import CalibrationProject, CameraConfig, PatternConfig, PatternType
from calibration.windshield.base import WindshieldConfig, WindshieldModelType, windshield_result_key
from calibration.windshield.baseline import calibrate_baseline
from calibration.windshield.spherical import calibrate_spherical
from calibration.windshield.spline import calibrate_spline
from calibration.validation import split_train_test
from tests._windshield_test_utils import (
    build_synthetic_spherical_windshield_dataset,
    build_synthetic_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
)
from calibration.types import CameraModelType


def _project_with_windshield_data() -> CalibrationProject:
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D, shear_k=0.02)
    camera_config = default_camera_config()
    config = WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)
    train_ids = [f.image_info.image_id for f in dataset.frames]
    result = calibrate_baseline(dataset, config, camera_config, train_ids, [])

    return CalibrationProject(
        project_name="windshield-test",
        camera_config=camera_config,
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=6, squares_y=5, square_size=0.04),
        windshield_config=config,
        windshield_dataset=dataset,
        windshield_results={WindshieldModelType.BASELINE: result},
    )


def test_windshield_fields_round_trip_through_dict():
    project = _project_with_windshield_data()

    payload = project_to_dict(project)
    restored = project_from_dict(payload)

    assert restored.windshield_config is not None
    assert restored.windshield_config.base_model_name == project.windshield_config.base_model_name
    assert (restored.windshield_config.base_camera_matrix == project.windshield_config.base_camera_matrix).all()
    assert (restored.windshield_config.base_distortion == project.windshield_config.base_distortion).all()

    assert restored.windshield_dataset is not None
    assert restored.windshield_dataset.num_total == project.windshield_dataset.num_total

    assert WindshieldModelType.BASELINE in restored.windshield_results
    restored_result = restored.windshield_results[WindshieldModelType.BASELINE]
    original_result = project.windshield_results[WindshieldModelType.BASELINE]
    assert restored_result.success == original_result.success
    assert restored_result.residual_stats.rmse == original_result.residual_stats.rmse
    assert restored_result.mean_dx == original_result.mean_dx
    assert restored_result.spatial_error_map is not None
    assert len(restored_result.spatial_error_map.cells) == len(original_result.spatial_error_map.cells)


def test_windshield_fields_round_trip_through_disk(tmp_path):
    project = _project_with_windshield_data()
    path = tmp_path / "windshield_project.ccproj"

    save_project(project, str(path))
    restored, missing = load_project(str(path))

    assert restored.windshield_config is not None
    assert WindshieldModelType.BASELINE in restored.windshield_results


def test_residual_grid_and_rbf_results_round_trip_as_separate_entries():
    project = _project_with_windshield_data()
    baseline_result = project.windshield_results[WindshieldModelType.BASELINE]
    grid_key = windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "grid")
    rbf_key = windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "rbf")
    grid_result = dataclasses.replace(
        baseline_result,
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        fitted_params={"residual_ray_method": 0.0, "runtime_param_count": 12.0},
    )
    rbf_result = dataclasses.replace(
        baseline_result,
        windshield_model=WindshieldModelType.RESIDUAL_RAY,
        fitted_params={
            "residual_ray_method": 1.0,
            "runtime_param_count": 24.0,
            "residual_value_param_count": 24.0,
            "serialized_numeric_value_count": 40.0,
        },
    )
    project.windshield_results = {grid_key: grid_result, rbf_key: rbf_result}

    payload = project_to_dict(project)
    assert set(payload["project"]["windshield_results"]) == {"residual_ray:grid", "residual_ray:rbf"}

    restored = project_from_dict(payload)

    assert set(restored.windshield_results) == {grid_key, rbf_key}
    assert restored.windshield_results[grid_key].fitted_params["residual_ray_method"] == 0.0
    assert restored.windshield_results[rbf_key].fitted_params["residual_ray_method"] == 1.0


def test_baseline_test_side_fields_round_trip():
    """STEP1 -> STEP2 backfill(test_regional_error 등)이 실제로 채워지고
    저장/복원되는지 확인 - Train/Test 분할이 있는 데이터셋으로 baseline을
    돌린다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D, shear_k=0.02)
    camera_config = default_camera_config()
    config = WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[: len(ids) - 2], ids[len(ids) - 2 :]
    result = calibrate_baseline(dataset, config, camera_config, train_ids, test_ids)
    assert result.test_regional_error is not None
    assert result.test_mean_dx is not None

    d = json_safe(dataclasses.asdict(result))
    restored = _windshield_calibration_result_from_dict(d)

    assert restored.test_regional_error is not None
    assert restored.test_mean_dx == result.test_mean_dx
    assert restored.test_mean_dy == result.test_mean_dy
    assert restored.ray_angular_error_deg == result.ray_angular_error_deg  # Baseline은 항상 None


def test_spherical_result_with_new_fields_round_trips():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPHERICAL,
        windshield_position_hint={"sphere_center_z": -8.0, "sphere_radius": 9.0},
    )
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    result = calibrate_spherical(dataset, config, camera_config, train_ids, test_ids)
    assert result.success

    project = CalibrationProject(
        project_name="spherical-test",
        camera_config=camera_config,
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=6, squares_y=5, square_size=0.04),
        windshield_config=config,
        windshield_dataset=dataset,
        windshield_results={WindshieldModelType.SPHERICAL: result},
    )
    restored = project_from_dict(project_to_dict(project))
    restored_result = restored.windshield_results[WindshieldModelType.SPHERICAL]

    assert restored_result.fitted_params["sphere_radius"] == pytest.approx(result.fitted_params["sphere_radius"])
    assert restored_result.ray_angular_error_deg == pytest.approx(result.ray_angular_error_deg)
    if result.test_ray_angular_error_deg is not None:
        assert restored_result.test_ray_angular_error_deg == pytest.approx(result.test_ray_angular_error_deg)


def test_windshield_result_missing_new_fields_loads_with_none_defaults():
    """새 8개 필드가 없는(STEP1 시절) 저장된 결과 dict도 그대로 로드되어야 한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)
    train_ids = [f.image_info.image_id for f in dataset.frames]
    result = calibrate_baseline(dataset, config, camera_config, train_ids, [])

    d = json_safe(dataclasses.asdict(result))
    for key in (
        "test_regional_error", "test_radial_profile", "test_radial_bands",
        "test_spatial_error_map", "test_mean_dx", "test_mean_dy",
        "ray_angular_error_deg", "test_ray_angular_error_deg",
    ):
        del d[key]

    restored = _windshield_calibration_result_from_dict(d)
    assert restored.test_regional_error is None
    assert restored.test_mean_dx is None
    assert restored.ray_angular_error_deg is None
    assert restored.test_ray_angular_error_deg is None


def test_residual_ray_hint_and_spline_hint_round_trip_through_dict():
    """이전 버그(STEP 3-A에서 residual_ray_hint 필드는 추가됐지만
    _windshield_config_from_dict()의 복원 코드가 빠져 있었음 - 프로젝트
    저장 후 재로드하면 method/AUTO/manual 설정이 사라졌다)의 회귀 테스트.
    spline_hint도 동일한 패턴이라 함께 확인한다."""
    K, D = default_camera_matrix_distortion()
    config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
        residual_ray_hint={"method": "rbf", "auto_rbf": 1.0},
        spline_hint={"auto_spline": 0.0, "spline_rows": 4.0, "spline_cols": 6.0},
    )
    project = CalibrationProject(
        project_name="hint-round-trip",
        camera_config=default_camera_config(),
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=6, squares_y=5, square_size=0.04),
        windshield_config=config,
    )

    restored = project_from_dict(project_to_dict(project))

    assert restored.windshield_config.residual_ray_hint == {"method": "rbf", "auto_rbf": 1.0}
    assert restored.windshield_config.spline_hint == {"auto_spline": 0.0, "spline_rows": 4.0, "spline_cols": 6.0}


def test_spline_result_round_trips_through_project():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = WindshieldConfig(
        base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPLINE, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0},
    )
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)
    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)
    assert result.success, result.error_message

    project = CalibrationProject(
        project_name="spline-test",
        camera_config=camera_config,
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=6, squares_y=5, square_size=0.04),
        windshield_config=config,
        windshield_dataset=dataset,
        windshield_results={WindshieldModelType.SPLINE: result},
    )
    restored = project_from_dict(project_to_dict(project))
    restored_result = restored.windshield_results[WindshieldModelType.SPLINE]

    assert restored_result.fitted_params["sphere_radius"] == pytest.approx(result.fitted_params["sphere_radius"])
    assert restored_result.fitted_params["spline_rows"] == result.fitted_params["spline_rows"]
    rows, cols = int(result.fitted_params["spline_rows"]), int(result.fitted_params["spline_cols"])
    for r in range(rows):
        for c in range(cols):
            key = f"spline_ds_{r}_{c}"
            assert restored_result.fitted_params[key] == pytest.approx(result.fitted_params[key])


def test_old_project_without_windshield_fields_loads_with_none_defaults():
    project = CalibrationProject(
        project_name="legacy",
        camera_config=CameraConfig(width=640, height=480),
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=5, squares_y=5, square_size=0.02),
    )
    payload = project_to_dict(project)
    # windshield 필드가 아예 없었던(이번 기능 이전) 프로젝트 파일을 흉내낸다.
    del payload["project"]["windshield_config"]
    del payload["project"]["windshield_dataset"]
    del payload["project"]["windshield_results"]
    assert payload["format_version"] == PROJECT_FORMAT_VERSION

    restored = project_from_dict(payload)

    assert restored.windshield_config is None
    assert restored.windshield_dataset is None
    assert restored.windshield_results == {}
