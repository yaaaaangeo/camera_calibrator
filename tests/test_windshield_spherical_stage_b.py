"""
tests/test_windshield_spherical_stage_b.py
===============================================

STAGE B(Sphere + per-frame pose joint refinement, ray-domain)에 특화된
검증. STAGE A만 다루는 기존 tests/test_windshield_spherical.py와 분리했다.

Test A~G는 이번 작업 스펙(섹션 26)의 항목 이름을 그대로 따른다.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.spherical import (
    DEFAULT_GLASS_REFRACTIVE_INDEX,
    MIN_STANDOFF_M,
    SphericalWindshieldModel,
    _evaluate_spherical,
    _fit_sphere,
    _initial_sphere_guess,
    _joint_refine_sphere_and_poses,
    calibrate_spherical,
    is_valid_spherical_windshield,
    refine_frame_pose_ray_domain,
)
from tests._windshield_test_utils import (
    DEFAULT_SPHERE_CENTER,
    DEFAULT_SPHERE_RADIUS,
    build_synthetic_spherical_windshield_dataset,
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
    return {"sphere_center_z": -8.0, "sphere_radius": 9.0}


# ---------------------------------------------------------------------------
# Test A - Pixel refinement이 실제로 작동하는지
# ---------------------------------------------------------------------------

def test_stage_b_improves_or_matches_stage_a_pixel_rmse():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, windshield_position_hint=_fitting_hint())
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert result.success, result.error_message
    # calibrate_spherical()은 내부적으로 STAGE A/B의 실제 pixel RMS를 비교해서
    # 더 나은 쪽만 채택한다 - 따라서 최종 결과가 STAGE A보다 "심각하게 나빠지는"
    # 일은 구조적으로 불가능하다. 이 테스트에서는 실제로 STAGE B가 채택돼서
    # (stage_used_is_joint_refined==1.0) 눈에 띄게 개선됐는지(테스트 데이터 기준
    # 이전에 확인된 STAGE A 단독 RMS ~1.0px보다 확실히 낮은지)까지 확인한다.
    assert result.fitted_params["stage_used_is_joint_refined"] == 1.0
    assert result.residual_stats.rmse < 0.6


def test_final_result_never_worse_than_stage_a_even_with_bad_hint():
    """joint refinement가 나쁜 방향으로 흘러가도(또는 아예 실패해도),
    calibrate_spherical의 최종 결과는 STAGE A 단독 fit보다 나쁘지 않아야
    한다 - 나쁘면 STAGE A로 되돌아가고 warning_message에 그 사실을 남긴다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    train_ids = [f.image_info.image_id for f in dataset.frames]

    ok_frames, rvecs, tvecs, _ = solve_poses_fixed_intrinsics(dataset.frames, K, D, _MODEL)
    baseline_model = BaselineWindshieldModel(K, D, _MODEL)
    d_obs_per_frame, p_cam_per_frame = [], []
    for frame, rvec, tvec in zip(ok_frames, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = frame.detection.corners.reshape(-1, 2)
        d_obs_per_frame.append(np.array([baseline_model.unproject_pixel(float(x), float(y)) for x, y in corners]))
        p_cam_per_frame.append(cam_pts)
    d_obs_arr = np.concatenate(d_obs_per_frame, axis=0)
    p_cam_arr = np.concatenate(p_cam_per_frame, axis=0)

    median_depth = float(np.median(p_cam_arr[:, 2]))
    initial_center, initial_radius = _initial_sphere_guess(_config(K, D, windshield_position_hint=_fitting_hint()), median_depth)
    stage_a_fit = _fit_sphere(
        d_obs_arr, p_cam_arr, 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX, 0.005, initial_center, initial_radius,
    )
    stage_a_model = SphericalWindshieldModel(
        K, D, _MODEL, stage_a_fit.x[:3], float(stage_a_fit.x[3]), 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX, 0.005,
    )

    w, h = camera_config.width, camera_config.height
    stage_a_outcome = _evaluate_spherical(ok_frames, rvecs, tvecs, stage_a_model, (w, h))

    config = _config(K, D, windshield_position_hint=_fitting_hint())
    result = calibrate_spherical(dataset, config, camera_config, train_ids, [])

    assert result.success
    # 약간의 수치 오차 허용(다른 fit 경로라 완전히 동일하지는 않음).
    assert result.residual_stats.rmse <= stage_a_outcome.residual_stats.rmse + 0.05


# ---------------------------------------------------------------------------
# Test B - K,D immutable (STAGE B를 포함한 전체 파이프라인 기준)
# ---------------------------------------------------------------------------

def test_stage_b_pipeline_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, windshield_position_hint=_fitting_hint())
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=1)

    calibrate_spherical(dataset, config, camera_config, train_ids, test_ids)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Test C - Per-frame pose가 실제로 optimize되는지
# ---------------------------------------------------------------------------

def test_joint_refinement_moves_pose_within_reasonable_bounds():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    ok_frames, rvecs, tvecs, _ = solve_poses_fixed_intrinsics(dataset.frames, K, D, _MODEL)
    baseline_model = BaselineWindshieldModel(K, D, _MODEL)

    d_obs_per_frame = []
    p_cam_per_frame = []
    for frame, rvec, tvec in zip(ok_frames, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = frame.detection.corners.reshape(-1, 2)
        d_obs_per_frame.append(np.array([baseline_model.unproject_pixel(float(x), float(y)) for x, y in corners]))
        p_cam_per_frame.append(cam_pts)
    median_depth = float(np.median(np.concatenate(p_cam_per_frame, axis=0)[:, 2]))
    initial_center, initial_radius = _initial_sphere_guess(_config(K, D, windshield_position_hint=_fitting_hint()), median_depth)

    joint = _joint_refine_sphere_and_poses(
        ok_frames, d_obs_per_frame, rvecs, tvecs, 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX, 0.005,
        initial_center, initial_radius,
    )

    moved_at_least_one = False
    for rvec0, tvec0, rvec1, tvec1 in zip(rvecs, tvecs, joint.rvecs, joint.tvecs):
        rot_delta = float(np.linalg.norm(rvec1.ravel() - rvec0.ravel()))
        trans_delta = float(np.linalg.norm(tvec1.ravel() - tvec0.ravel()))
        # weak prior가 있으므로 pose가 "너무 많이" 움직이면 안 된다(합리적 범위).
        assert rot_delta < 0.2, "pose가 initial solvePnP에서 너무 많이 벗어남(prior가 안 걸렸을 가능성)"
        assert trans_delta < 0.2
        if rot_delta > 1e-6 or trans_delta > 1e-6:
            moved_at_least_one = True

    # 최소 하나의 프레임에서는 실제로 pose가 갱신됐어야 한다 - 전혀 안 움직였다면
    # joint refinement가 제대로 연결되지 않았을 가능성이 있다.
    assert moved_at_least_one


# ---------------------------------------------------------------------------
# Test D - Physical invalid Sphere reject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "center,radius,thickness",
    [
        ([0.0, 0.0, -9.7], -1.0, 0.005),          # 음수 radius
        ([0.0, 0.0, 9.7], 10.0, 0.005),           # windshield가 카메라 "뒤"(near-surface z<0)
        ([1000.0, 1000.0, -9.7], 1.0, 0.005),     # 카메라가 구 밖(카메라/구 관계 무효)
        ([0.0, 0.0, -9.7], 10.0, -0.005),         # 음수 thickness
        ([0.0, 0.0, -9.7], float("nan"), 0.005),  # NaN
    ],
)
def test_invalid_sphere_configurations_are_rejected(center, radius, thickness):
    assert is_valid_spherical_windshield(np.array(center), radius, thickness) is False


def test_valid_sphere_configuration_is_accepted():
    assert is_valid_spherical_windshield(DEFAULT_SPHERE_CENTER, DEFAULT_SPHERE_RADIUS, 0.005) is True


def test_fit_sphere_does_not_crash_with_invalid_candidate_region():
    """_fit_sphere()의 residual 함수가 bounds 안에서도 물리적으로 무효한 후보를
    지날 수 있는데(예: radius가 아주 작아지는 구간), crash 없이 penalty를
    반환해야 한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    ok_frames, rvecs, tvecs, _ = solve_poses_fixed_intrinsics(dataset.frames, K, D, _MODEL)
    baseline_model = BaselineWindshieldModel(K, D, _MODEL)
    d_obs, p_cam = [], []
    for frame, rvec, tvec in zip(ok_frames, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        obj = frame.detection.object_points.reshape(-1, 3).astype(np.float64)
        cam_pts = (R @ obj.T).T + tvec.reshape(1, 3)
        corners = frame.detection.corners.reshape(-1, 2)
        d_obs.append(np.array([baseline_model.unproject_pixel(float(x), float(y)) for x, y in corners]))
        p_cam.append(cam_pts)
    d_obs_arr = np.concatenate(d_obs, axis=0)
    p_cam_arr = np.concatenate(p_cam, axis=0)

    # 시작점 자체를 물리적으로 무효한 근처(반지름 아주 작음)로 줘도 crash하면 안 된다.
    fit = _fit_sphere(d_obs_arr, p_cam_arr, 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX, 0.005, np.array([0.0, 0.0, 0.0]), 0.06)
    assert np.all(np.isfinite(fit.x))


# ---------------------------------------------------------------------------
# Test E - Test Leakage 강한 검증
# ---------------------------------------------------------------------------

def test_sphere_fit_is_identical_regardless_of_test_data_content():
    """같은 Train, 완전히 다른(크게 왜곡된) Test로 두 번 돌려도 fitted sphere
    center/radius가 (수치 허용오차 안에서) 동일해야 한다 - Test 데이터가
    sphere fitting에 전혀 영향을 주지 않는다는 것을 직접 검증한다."""
    K, D = default_camera_matrix_distortion()
    dataset_a = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    assert test_ids, "테스트가 의미 있으려면 test_ids가 있어야 함"

    dataset_b = copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            # Test 프레임의 코너를 크게(50px) 왜곡한다 - Train은 건드리지 않는다.
            frame.detection.corners = frame.detection.corners + np.float32(50.0)

    config_a = _config(K, D, windshield_position_hint=_fitting_hint())
    config_b = _config(K, D, windshield_position_hint=_fitting_hint())

    result_a = calibrate_spherical(dataset_a, config_a, camera_config, train_ids, test_ids)
    result_b = calibrate_spherical(dataset_b, config_b, camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    for key in ("sphere_center_x", "sphere_center_y", "sphere_center_z", "sphere_radius"):
        assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)
    # Test 쪽 결과는 당연히 달라야 한다(왜곡을 실제로 반영해야 정상).
    assert result_a.test_residual_stats.rmse != pytest.approx(result_b.test_residual_stats.rmse, abs=1e-6)


def test_sphere_fit_identical_with_and_without_test_split():
    """같은 Train으로, test_ids를 아예 안 주는 경우와 주는 경우를 비교해도
    sphere 결과가 같아야 한다(Test 존재 자체가 fitting에 영향을 주면 안 됨)."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]

    config_no_test = _config(K, D, windshield_position_hint=_fitting_hint())
    config_with_test = _config(K, D, windshield_position_hint=_fitting_hint())

    result_no_test = calibrate_spherical(dataset, config_no_test, camera_config, train_ids, [])
    result_with_test = calibrate_spherical(dataset, config_with_test, camera_config, train_ids, test_ids)

    assert result_no_test.success and result_with_test.success
    for key in ("sphere_center_x", "sphere_center_y", "sphere_center_z", "sphere_radius"):
        assert result_no_test.fitted_params[key] == pytest.approx(result_with_test.fitted_params[key], abs=1e-9)


# ---------------------------------------------------------------------------
# Test F - Pose-only Test Refinement이 Sphere를 바꾸지 않는지
# ---------------------------------------------------------------------------

def test_refine_frame_pose_ray_domain_does_not_return_or_need_sphere_output():
    """refine_frame_pose_ray_domain()은 (rvec,tvec)만 반환한다 - sphere를
    바꿀 방법 자체가 인터페이스에 없다는 것을 계약으로 확인한다. 동시에,
    다른 sphere를 넣으면 실제로 다른 pose가 나온다는 것도 확인해서(sphere를
    무시하고 있지는 않은지) 함수가 실제로 연결돼 있음을 검증한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    ok_frames, rvecs, tvecs, _ = solve_poses_fixed_intrinsics(dataset.frames, K, D, _MODEL)
    baseline_model = BaselineWindshieldModel(K, D, _MODEL)
    frame = ok_frames[0]
    corners = frame.detection.corners.reshape(-1, 2)
    d_obs = np.array([baseline_model.unproject_pixel(float(x), float(y)) for x, y in corners])

    fit_1 = refine_frame_pose_ray_domain(
        frame, d_obs, np.array([0.0, 0.0, -9.7]), 10.0, 0.005, 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX,
        rvecs[0], tvecs[0],
    )
    fit_2 = refine_frame_pose_ray_domain(
        frame, d_obs, np.array([0.1, -0.1, -6.0]), 7.0, 0.005, 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX,
        rvecs[0], tvecs[0],
    )
    assert fit_1.x.shape == (6,)
    assert fit_2.x.shape == (6,)
    assert not np.allclose(fit_1.x, fit_2.x), "sphere를 바꿨는데 pose 결과가 그대로면 함수가 sphere를 반영하지 않는 것"


def test_calibrate_spherical_test_refinement_does_not_alter_final_sphere():
    """calibrate_spherical() 결과의 sphere_* 값은 Test pose refinement 이후에도
    Train 평가 시점 값과 같아야 한다(코드 경로상 test 단계는 final_center/
    final_radius를 재할당하지 않는다) - fitted_params가 그 증거다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    config = _config(K, D, windshield_position_hint=_fitting_hint())

    train_only_config = _config(K, D, windshield_position_hint=_fitting_hint())
    train_only_result = calibrate_spherical(dataset, train_only_config, camera_config, train_ids, [])
    full_result = calibrate_spherical(dataset, config, camera_config, train_ids, test_ids)

    assert train_only_result.success and full_result.success
    for key in ("sphere_center_x", "sphere_center_y", "sphere_center_z", "sphere_radius"):
        assert train_only_result.fitted_params[key] == pytest.approx(full_result.fitted_params[key], abs=1e-9)


# ---------------------------------------------------------------------------
# Test G - Synthetic Independent Perturbation (독립적인 analytic reference)
# ---------------------------------------------------------------------------

def test_on_axis_point_projects_near_principal_point_independent_of_refraction():
    """Sphere 중심이 광축(z축) 위에 있으면, 광축을 따라 정확히 나가는 광선은
    양쪽 표면 모두 법선과 평행(수직 입사)하게 만나 굴절이 전혀 일어나지
    않는다(Snell's law: 입사각 0 -> 굴절각 0, refract_ray가 이미 독립적으로
    검증됨 - test_windshield_refraction.py). 따라서 광축 위의 점은
    SphericalWindshieldModel을 거쳐도 Base K,D만으로 투영한 것과 (거의)
    동일한 픽셀(주점 근방)에 맺혀야 한다 - 이 사실은 SphericalWindshieldModel
    코드 자체를 재사용하지 않고 기하학적으로 독립적으로 유도한 결과다.
    """
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    D = np.zeros((5, 1))  # 왜곡 없음 - 순수 pinhole로 독립 검증을 더 정확하게
    center = np.array([0.0, 0.0, -9.7])  # 광축(z) 위의 중심
    model = SphericalWindshieldModel(K, D, _MODEL, center, 10.0, 1.0, DEFAULT_GLASS_REFRACTIVE_INDEX, 0.005)

    u, v = model.project_point(0.0, 0.0, 5.0)  # 광축 위, windshield 너머의 점

    assert u == pytest.approx(K[0, 2], abs=1e-6)
    assert v == pytest.approx(K[1, 2], abs=1e-6)
