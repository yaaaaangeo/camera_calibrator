"""
tests/test_windshield_spherical.py
=======================================

Phase 2(Spherical) 핵심 검증.

중요한 배경: 카메라 하나의 좁은 화각으로 큰 반지름의 구(windshield)를
관측해서 그 중심/반지름을 복원하는 문제는 본질적으로 ill-conditioned하다 -
narrow-FOV에서는 "실제 큰 구"와 "그 국소 곡률만 비슷한 다른 구"가 관측
데이터상 거의 구분되지 않는다(단일 카메라 자기-캘리브레이션 문헌에서 잘
알려진 현상). 그래서 이 테스트는 "정확한 sphere_center/radius를 복원하는가"
보다 "optimizer가 실제로 수렴해서 각도 정렬 잔차를 0에 가깝게 만드는가"를
1차 정답 기준으로 삼고, 기하 복원은 느슨한 허용 오차로만 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import calibrate_baseline
from calibration.windshield.projection import build_projector
from calibration.windshield.spherical import (
    DEFAULT_AIR_REFRACTIVE_INDEX,
    MAX_ACCEPTABLE_CORNER_FAILURE_RATE,
    SphericalWindshieldModel,
    calibrate_spherical,
)
from tests._windshield_test_utils import (
    DEFAULT_GLASS_INDEX,
    DEFAULT_GLASS_THICKNESS_M,
    DEFAULT_SPHERE_CENTER,
    DEFAULT_SPHERE_RADIUS,
    build_synthetic_spherical_windshield_dataset,
    build_synthetic_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _config(K, D, **kwargs) -> WindshieldConfig:
    return WindshieldConfig(
        base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPHERICAL, **kwargs,
    )


def _fitting_hint() -> dict:
    """실제 정답에 가까운(하지만 정확히 같지는 않은) initial hint -
    최적화가 합리적인 시간 안에 수렴하도록 돕는다. 정답을 그대로 주지
    않는다 - 그러면 최적화가 사실상 아무 일도 안 하는 것이 된다."""
    return {"sphere_center_z": -8.0, "sphere_radius": 9.0}


def test_spherical_constructor_and_build_projector_round_trip():
    K, D = default_camera_matrix_distortion()
    from calibration.windshield.base import WindshieldCalibrationResult

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.SPHERICAL,
        base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D,
        fitted_params={
            "sphere_center_x": 0.0, "sphere_center_y": 0.0, "sphere_center_z": -9.7,
            "sphere_radius": 10.0, "air_refractive_index": 1.0, "glass_refractive_index": 1.52,
            "glass_thickness_m": 0.005,
        },
        success=True,
    )
    model = build_projector(result)
    assert isinstance(model, SphericalWindshieldModel)
    # 굴절이 실제로 project_point에 영향을 준다 - Baseline과 다른 픽셀이 나와야 한다.
    from calibration.windshield.baseline import BaselineWindshieldModel

    baseline = BaselineWindshieldModel(K, D, _MODEL)
    u1, v1 = model.project_point(0.3, 0.2, 5.0)
    u2, v2 = baseline.project_point(0.3, 0.2, 5.0)
    assert (u1, v1) != pytest.approx((u2, v2), abs=1e-6)


def test_calibrate_spherical_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, windshield_position_hint=_fitting_hint())
    train_ids = [f.image_info.image_id for f in dataset.frames]

    calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_calibrate_spherical_converges_to_low_angular_residual():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    assert len(dataset.frames) >= 3, "합성 데이터셋 생성 자체가 실패하면 안 됨"
    camera_config = default_camera_config()
    config = _config(K, D, windshield_position_hint=_fitting_hint())
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert result.success, result.error_message
    # optimizer_cost는 least_squares의 0.5*sum(residual^2) - 각 residual은
    # 두 단위벡터 차이(대략 라디안 각도 오차)이므로, 이 값이 작다는 것은
    # 실제로 광선이 잘 정렬됐다는 뜻이다(픽셀 RMS는 초점거리로 증폭되므로
    # 여기서는 1차 신호로 안 쓴다).
    assert result.fitted_params["optimizer_cost"] < 0.01
    assert result.ray_angular_error_deg is not None
    assert result.ray_angular_error_deg < 1.0  # degrees


def test_calibrate_spherical_recovers_approximate_sphere_geometry():
    """느슨한 허용 오차(narrow-FOV ill-conditioning을 고려) - 정답의 절반~두 배
    수준으로만 확인한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, windshield_position_hint=_fitting_hint())
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert result.success
    fitted_radius = result.fitted_params["sphere_radius"]
    assert DEFAULT_SPHERE_RADIUS * 0.5 < fitted_radius < DEFAULT_SPHERE_RADIUS * 2.0

    fitted_center = np.array([
        result.fitted_params["sphere_center_x"],
        result.fitted_params["sphere_center_y"],
        result.fitted_params["sphere_center_z"],
    ])
    assert np.linalg.norm(fitted_center - DEFAULT_SPHERE_CENTER) < DEFAULT_SPHERE_RADIUS


def test_calibrate_spherical_zero_refraction_matches_baseline():
    """n_air == n_glass면 굴절이 전혀 없으므로, 어떤 sphere를 골라도 residual은
    Baseline과 거의 같아야 한다(사용자 스펙 28-7 Zero-refraction sanity)."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)  # Baseline용 - 굴절 없이 생성됨
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames]

    baseline_config = WindshieldConfig(base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D)
    baseline_result = calibrate_baseline(dataset, baseline_config, camera_config, train_ids, [])

    spherical_config = _config(
        K, D, glass_refractive_index=DEFAULT_AIR_REFRACTIVE_INDEX,
        windshield_position_hint={"sphere_center_z": -5.0, "sphere_radius": 6.0},
    )
    spherical_result = calibrate_spherical(dataset, spherical_config, camera_config, train_ids, [])

    assert spherical_result.success
    assert spherical_result.residual_stats.rmse < baseline_result.residual_stats.rmse + 0.05


def test_calibrate_spherical_train_test_split_has_no_leakage():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, windshield_position_hint=_fitting_hint())

    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)
    assert train_ids and test_ids
    assert set(train_ids).isdisjoint(test_ids)

    result = calibrate_spherical(dataset, config, camera_config, train_ids, test_ids)

    assert result.train_frame_ids == train_ids
    assert result.test_frame_ids == test_ids
    if result.success:
        assert result.residual_stats.n > 0
        # Test 쪽 필드가 실제로 채워지는지(STEP1 backfill과 대칭) 확인.
        assert result.test_residual_stats is not None


def test_calibrate_spherical_too_few_train_frames_returns_failure_not_exception():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)

    result = calibrate_spherical(dataset, config, camera_config, [], [])

    assert result.success is False
    assert result.error_message


def test_calibrate_spherical_too_few_corners_returns_failure():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)
    # 프레임 하나만 train으로 주면(MIN_FRAMES_REQUIRED=3 미달) 프레임 부족으로 실패.
    train_ids = [dataset.frames[0].image_info.image_id]

    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert result.success is False
    assert result.error_message


def test_ray_angular_error_none_on_geometric_failure_finite_on_success():
    K, D = default_camera_matrix_distortion()
    model = SphericalWindshieldModel(K, D, _MODEL, DEFAULT_SPHERE_CENTER, DEFAULT_SPHERE_RADIUS)

    # 성공 케이스 - 카메라 광축 위, windshield 너머의 점.
    angle = model.ray_angular_error_deg(K[0, 2], K[1, 2], np.array([0.0, 0.0, 5.0]))
    assert angle is not None
    assert angle >= 0.0

    # 실패 케이스 - 구를 완전히 빗나가는 픽셀(광축에서 아주 멀리 떨어진 지점을
    # 광선 방향으로 써서 sphere를 비껴가게 만든다).
    far_center = np.array([1000.0, 1000.0, -9.7])
    model_far = SphericalWindshieldModel(K, D, _MODEL, far_center, 1.0)
    angle_fail = model_far.ray_angular_error_deg(K[0, 2], K[1, 2], np.array([0.0, 0.0, 5.0]))
    assert angle_fail is None


def test_evaluate_spherical_counts_projection_failures_with_bad_sphere():
    """optimizer 결과와 무관하게, _evaluate_spherical 자체의 실패 카운팅
    로직을 결정론적으로 검증한다 - sphere를 광축에서 완전히 벗어난 곳에
    작게 두면 거의 모든 코너의 광선이 교차하지 않아야 한다."""
    from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
    from calibration.windshield.spherical import _evaluate_spherical

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    ok_frames, rvecs, tvecs, _ = solve_poses_fixed_intrinsics(dataset.frames, K, D, _MODEL)

    bad_model = SphericalWindshieldModel(
        K, D, _MODEL, sphere_center=np.array([500.0, 500.0, -1.0]), sphere_radius=0.1
    )
    outcome = _evaluate_spherical(ok_frames, rvecs, tvecs, bad_model, (K[0, 2] * 2, K[1, 2] * 2))

    total = outcome.num_points_ok + outcome.num_points_failed
    assert total > 0
    failure_rate = outcome.num_points_failed / total
    assert failure_rate > MAX_ACCEPTABLE_CORNER_FAILURE_RATE


def test_calibrate_spherical_returns_clean_result_with_unreachable_hint():
    """windshield_position_hint가 완전히 비현실적이어도 calibrate_spherical은
    crash하지 않고 명확한 success=True/False 중 하나로 끝나야 한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    bad_hint = {"sphere_center_x": 500.0, "sphere_center_y": 500.0, "sphere_center_z": -1.0, "sphere_radius": 0.1}
    config = _config(K, D, windshield_position_hint=bad_hint)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert isinstance(result.success, bool)
    if not result.success:
        assert result.error_message
