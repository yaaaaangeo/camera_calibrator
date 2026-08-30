"""
Tests for camera_lidar.guided_roi (pure geometry helpers) and its
orchestration inside camera_lidar.pipeline._detect_lidar_guided /
calibrate_single_scene (GUIDED AUTO ROI mode).

Load-bearing checks, per the GUIDED ROI design spec:
  - the Targetless prior only ever narrows WHICH POINTS reach the existing
    multi-plane LiDAR detector (never the final correspondence/solve);
  - the margin schedule actually expands and is tried smallest-first;
  - fallback to full-cloud AUTO only happens when every guided attempt
    failed AND config.fallback_to_auto is set;
  - a PARTIAL camera scene (3 real markers) still predicts an ROI from all
    4 pose-inferred circle centers, while the final correspondence still
    only uses the 3 actually-detected ids.
"""

from __future__ import annotations

import numpy as np
import pytest

import camera_lidar.pipeline as pipeline
from camera_lidar.camera_detector import CameraDetectionResult
from camera_lidar.guided_roi import (
    build_margin_schedule,
    build_roi_from_predicted_centers,
    compute_guided_base_margin,
    predict_circle_centers_lidar,
)
from camera_lidar.lidar_detector import LidarDetectionResult
from camera_lidar.pipeline import calibrate_single_scene
from camera_lidar.target_config import CORNER_ORDER, TargetConfig
from camera_lidar.types import (
    CalibrationScene,
    FailureReason,
    GuidedROIConfig,
    ImageFrame,
    PointCloudFrame,
    ROIConfig,
    SceneType,
    TargetlessPrior,
)
from geometry.transform import invert_transform, rpy_to_rotation_matrix, to_homogeneous


def _identity_prior() -> TargetlessPrior:
    return TargetlessPrior(T_lidar_from_camera=np.eye(4), source_path="calib.json", source_key="T_lidar_camera")


def _make_scene(guided_config) -> CalibrationScene:
    target = TargetConfig()
    image = ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    cloud = PointCloudFrame(timestamp=0.0, points=np.zeros((10, 3)))
    return CalibrationScene(image=image, cloud=cloud, intrinsics=None, target=target, guided_roi=guided_config)


# ---------------------------------------------------------------------------
# TEST 1 -- camera -> lidar center prediction
# ---------------------------------------------------------------------------

def test_predict_circle_centers_lidar_applies_prior():
    T = to_homogeneous(np.eye(3), np.array([1.0, 0.0, 0.0]))
    camera_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    predicted = predict_circle_centers_lidar(camera_centers, T)
    assert np.allclose(predicted, camera_centers + np.array([1.0, 0.0, 0.0]))


def test_predict_circle_centers_lidar_rejects_bad_shapes():
    with pytest.raises(ValueError):
        predict_circle_centers_lidar(np.zeros((3, 3)), np.eye(4))
    with pytest.raises(ValueError):
        predict_circle_centers_lidar(np.zeros((4, 3)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# TEST 2 -- AABB contains all 4 predicted centers
# ---------------------------------------------------------------------------

def test_build_roi_from_predicted_centers_contains_all_points():
    centers = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 5.0, 3.0],
        [1.0, 2.0, -3.0],
        [4.0, -2.0, 0.0],
    ])
    roi = build_roi_from_predicted_centers(centers, margin_m=0.5)
    assert roi.x_min <= centers[:, 0].min() and roi.x_max >= centers[:, 0].max()
    assert roi.y_min <= centers[:, 1].min() and roi.y_max >= centers[:, 1].max()
    assert roi.z_min <= centers[:, 2].min() and roi.z_max >= centers[:, 2].max()
    # margin actually applied, not just coincidentally containing the points
    assert roi.x_min == pytest.approx(centers[:, 0].min() - 0.5)
    assert roi.x_max == pytest.approx(centers[:, 0].max() + 0.5)


# ---------------------------------------------------------------------------
# TEST 3 -- margin grows with predicted range (rotation uncertainty term)
# ---------------------------------------------------------------------------

def test_margin_increases_with_distance():
    config = GuidedROIConfig(prior=_identity_prior(), min_margin_m=0.001, max_margin_m=100.0)
    near = np.tile([0.0, 0.0, 1.0], (4, 1))
    far = np.tile([0.0, 0.0, 5.0], (4, 1))

    near_margin, near_dist = compute_guided_base_margin(near, config)
    far_margin, far_dist = compute_guided_base_margin(far, config)

    assert far_dist > near_dist
    assert far_margin > near_margin


# ---------------------------------------------------------------------------
# TEST 4 -- margin clipped to [min_margin_m, max_margin_m]
# ---------------------------------------------------------------------------

def test_margin_respects_min_and_max_clip():
    tiny_uncertainty_config = GuidedROIConfig(
        prior=_identity_prior(),
        translation_uncertainty_m=0.0, rotation_uncertainty_deg=0.0, safety_margin_m=0.0,
        min_margin_m=0.4, max_margin_m=2.0,
    )
    centers = np.tile([0.0, 0.0, 1.0], (4, 1))
    margin, _ = compute_guided_base_margin(centers, tiny_uncertainty_config)
    assert margin == pytest.approx(0.4)

    huge_uncertainty_config = GuidedROIConfig(
        prior=_identity_prior(),
        translation_uncertainty_m=100.0, rotation_uncertainty_deg=0.0, safety_margin_m=0.0,
        min_margin_m=0.4, max_margin_m=2.0,
    )
    margin, _ = compute_guided_base_margin(centers, huge_uncertainty_config)
    assert margin == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TEST 5 -- margin schedule is ascending, unique, and capped at max_margin_m
# ---------------------------------------------------------------------------

def test_margin_schedule_ascending_unique_capped():
    config = GuidedROIConfig(
        prior=_identity_prior(), max_margin_m=1.2, expansion_factors=(1.0, 1.5, 2.0),
    )
    schedule = build_margin_schedule(0.6, config)
    assert schedule == sorted(schedule)
    assert len(schedule) == len(set(round(m, 6) for m in schedule))
    assert all(m <= config.max_margin_m for m in schedule)
    assert schedule[0] == pytest.approx(0.6)
    assert schedule[-1] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# GuidedROIConfig.__post_init__ validation (hardening): tan() is not a sane
# uncertainty model near/beyond 90 degrees, so rotation_uncertainty_deg is
# capped at 30 degrees; margin bounds must be positive and ordered.
# Construction-time ValueError, never silent correction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotation_uncertainty_deg", [0.0, 5.0, 30.0])
def test_guided_roi_config_accepts_boundary_rotation_values(rotation_uncertainty_deg):
    config = GuidedROIConfig(prior=_identity_prior(), rotation_uncertainty_deg=rotation_uncertainty_deg)
    assert config.rotation_uncertainty_deg == rotation_uncertainty_deg


def test_guided_roi_config_accepts_valid_margin_bounds():
    config = GuidedROIConfig(prior=_identity_prior(), min_margin_m=0.4, max_margin_m=2.0)
    assert config.min_margin_m == 0.4
    assert config.max_margin_m == 2.0


@pytest.mark.parametrize("rotation_uncertainty_deg", [-0.1, 30.1, 45.0, 90.0, -10.0])
def test_guided_roi_config_rejects_out_of_range_rotation(rotation_uncertainty_deg):
    with pytest.raises(ValueError):
        GuidedROIConfig(prior=_identity_prior(), rotation_uncertainty_deg=rotation_uncertainty_deg)


def test_guided_roi_config_rejects_negative_translation_uncertainty():
    with pytest.raises(ValueError):
        GuidedROIConfig(prior=_identity_prior(), translation_uncertainty_m=-0.01)


def test_guided_roi_config_rejects_negative_safety_margin():
    with pytest.raises(ValueError):
        GuidedROIConfig(prior=_identity_prior(), safety_margin_m=-0.01)


@pytest.mark.parametrize("min_margin_m", [0.0, -0.5])
def test_guided_roi_config_rejects_non_positive_min_margin(min_margin_m):
    with pytest.raises(ValueError):
        GuidedROIConfig(prior=_identity_prior(), min_margin_m=min_margin_m, max_margin_m=2.0)


@pytest.mark.parametrize("max_margin_m", [0.0, -0.5])
def test_guided_roi_config_rejects_non_positive_max_margin(max_margin_m):
    with pytest.raises(ValueError):
        GuidedROIConfig(prior=_identity_prior(), max_margin_m=max_margin_m)


def test_guided_roi_config_rejects_min_greater_than_max():
    with pytest.raises(ValueError):
        GuidedROIConfig(prior=_identity_prior(), min_margin_m=2.0, max_margin_m=0.4)


# ---------------------------------------------------------------------------
# Pipeline-level orchestration tests (6-10): monkeypatch the LiDAR/camera
# detector entry points that camera_lidar.pipeline calls, so these test
# _detect_lidar_guided's/calibrate_single_scene's own control flow rather
# than re-running real plane/circle detection.
# ---------------------------------------------------------------------------

def _fake_camera_result(detected_ids=frozenset(CORNER_ORDER)) -> CameraDetectionResult:
    return CameraDetectionResult(
        success=True,
        circle_centers=np.array([[0.0, 0.0, 5.0]] * 4) + np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        detected_ids=detected_ids,
        markers_detected=len(detected_ids),
    )


def test_first_guided_roi_success_skips_full_cloud_auto(monkeypatch):
    config = GuidedROIConfig(prior=_identity_prior())
    scene = _make_scene(config)
    camera_result = _fake_camera_result()

    calls = []

    def fake_auto(cloud, target, max_planes=6, rng_seed=42, cancel_check=None, on_plane_candidate=None, search_roi=None):
        calls.append(search_roi)
        return LidarDetectionResult(success=True, circle_centers=camera_result.circle_centers)

    monkeypatch.setattr(pipeline, "detect_lidar_target_auto", fake_auto)

    result, diagnostics = pipeline._detect_lidar_guided(scene, camera_result)

    assert result.success
    assert diagnostics.fallback_to_auto_used is False
    # Every call so far must have been a restricted (non-None) search_roi --
    # full-cloud AUTO (search_roi=None) was never invoked.
    assert all(roi is not None for roi in calls)
    assert len(calls) == 1


def test_second_guided_roi_success_records_correct_margin(monkeypatch):
    config = GuidedROIConfig(prior=_identity_prior())
    scene = _make_scene(config)
    camera_result = _fake_camera_result()

    call_count = {"n": 0}

    def fake_auto(cloud, target, max_planes=6, rng_seed=42, cancel_check=None, on_plane_candidate=None, search_roi=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LidarDetectionResult(success=False, failure_reason=FailureReason.CIRCLES_NOT_FOUND)
        return LidarDetectionResult(success=True, circle_centers=camera_result.circle_centers)

    monkeypatch.setattr(pipeline, "detect_lidar_target_auto", fake_auto)

    result, diagnostics = pipeline._detect_lidar_guided(scene, camera_result)

    assert result.success
    assert diagnostics.fallback_to_auto_used is False
    assert len(diagnostics.attempted_margins_m) == 2
    assert diagnostics.selected_margin_m == diagnostics.attempted_margins_m[1]


def test_all_guided_attempts_fail_falls_back_to_full_auto_when_enabled(monkeypatch):
    config = GuidedROIConfig(prior=_identity_prior(), fallback_to_auto=True)
    scene = _make_scene(config)
    camera_result = _fake_camera_result()

    calls = []

    def fake_auto(cloud, target, max_planes=6, rng_seed=42, cancel_check=None, on_plane_candidate=None, search_roi=None):
        calls.append(search_roi)
        if search_roi is None:
            return LidarDetectionResult(success=True, circle_centers=camera_result.circle_centers)
        return LidarDetectionResult(success=False, failure_reason=FailureReason.CIRCLES_NOT_FOUND)

    monkeypatch.setattr(pipeline, "detect_lidar_target_auto", fake_auto)

    result, diagnostics = pipeline._detect_lidar_guided(scene, camera_result)

    assert result.success
    assert diagnostics.fallback_to_auto_used is True
    assert calls[-1] is None  # the final call was the unrestricted full-cloud AUTO


def test_all_guided_attempts_fail_no_fallback_returns_last_guided_failure(monkeypatch):
    config = GuidedROIConfig(prior=_identity_prior(), fallback_to_auto=False)
    scene = _make_scene(config)
    camera_result = _fake_camera_result()

    calls = []

    def fake_auto(cloud, target, max_planes=6, rng_seed=42, cancel_check=None, on_plane_candidate=None, search_roi=None):
        calls.append(search_roi)
        return LidarDetectionResult(success=False, failure_reason=FailureReason.CIRCLES_NOT_FOUND)

    monkeypatch.setattr(pipeline, "detect_lidar_target_auto", fake_auto)

    result, diagnostics = pipeline._detect_lidar_guided(scene, camera_result)

    assert result.success is False
    assert diagnostics.fallback_to_auto_used is False
    assert all(roi is not None for roi in calls)  # full-cloud AUTO (None) never called


# ---------------------------------------------------------------------------
# TEST 10 -- Targetless prior never reaches match_partial_centers as
# reference_transform.
# ---------------------------------------------------------------------------

def test_targetless_prior_never_becomes_reference_transform(monkeypatch):
    config = GuidedROIConfig(prior=_identity_prior())
    scene = _make_scene(config)
    camera_result = _fake_camera_result()

    monkeypatch.setattr(pipeline, "detect_camera_target", lambda *a, **k: camera_result)
    monkeypatch.setattr(
        pipeline, "detect_lidar_target_auto",
        lambda *a, **k: LidarDetectionResult(success=True, circle_centers=camera_result.circle_centers),
    )

    captured_kwargs = {}
    real_match = pipeline.match_partial_centers

    def spy_match(camera_ids_to_centers, lidar_centers_cyclic, reference_transform=None):
        captured_kwargs["reference_transform"] = reference_transform
        return real_match(camera_ids_to_centers, lidar_centers_cyclic, reference_transform=reference_transform)

    monkeypatch.setattr(pipeline, "match_partial_centers", spy_match)

    result = calibrate_single_scene(scene, roi_mode="guided", reference_transform=None)

    assert result.success
    # The prior's rotation/translation must NEVER show up as reference_transform --
    # only an explicit caller-supplied reference_transform may.
    assert captured_kwargs["reference_transform"] is None


# ---------------------------------------------------------------------------
# TEST 11 -- PARTIAL camera scene: ROI prediction uses all 4 pose-inferred
# centers, final correspondence uses only the 3 actually-detected ids.
# ---------------------------------------------------------------------------

def test_partial_camera_scene_uses_4_for_roi_3_for_correspondence(monkeypatch):
    target = TargetConfig()
    board_centers = target.circle_centers_board_frame()  # (4,3), CORNER_ORDER

    R_true = rpy_to_rotation_matrix(5, -3, 90, degrees=True)
    t_true = np.array([0.1, -0.2, 2.0])
    camera_centers = (R_true @ board_centers.T).T + t_true  # all 4, pose-inferred

    # Camera only actually detected 3 markers (missing bottom_left).
    detected_ids = frozenset(CORNER_ORDER) - {"bottom_left"}
    camera_result = CameraDetectionResult(
        success=True, circle_centers=camera_centers, detected_ids=detected_ids, markers_detected=3,
    )

    # LiDAR sees all 4 physical circles regardless of which markers the
    # camera identified -- use the board centers directly as "lidar frame".
    lidar_centers = board_centers.copy()

    config = GuidedROIConfig(prior=_identity_prior())
    scene = _make_scene(config)

    monkeypatch.setattr(pipeline, "detect_camera_target", lambda *a, **k: camera_result)

    captured_predicted = {}
    real_predict = pipeline.predict_circle_centers_lidar

    def spy_predict(camera_circle_centers, T):
        predicted = real_predict(camera_circle_centers, T)
        captured_predicted["shape"] = predicted.shape
        return predicted

    monkeypatch.setattr(pipeline, "predict_circle_centers_lidar", spy_predict)
    monkeypatch.setattr(
        pipeline, "detect_lidar_target_auto",
        lambda *a, **k: LidarDetectionResult(success=True, circle_centers=lidar_centers),
    )

    result = calibrate_single_scene(scene, roi_mode="guided", reference_transform=(R_true, t_true))

    assert result.success
    # ROI prediction used all 4 pose-inferred centers.
    assert captured_predicted["shape"] == (4, 3)
    # Final correspondence only used the 3 actually-detected ids.
    assert len(result.common_ids) == 3
    assert "bottom_left" not in result.common_ids
    assert result.scene_type == SceneType.VALID_PARTIAL
    assert result.camera_centers.shape == (3, 3)


# ---------------------------------------------------------------------------
# TRUE end-to-end integration: real detect_lidar_target_auto (real RANSAC
# plane search + real boundary/circle fit) against a real synthetic point
# cloud, driven entirely through calibrate_single_scene(roi_mode="guided").
# Everything in tests 1-11 above mocks detect_lidar_target_auto itself, so
# it only proves _detect_lidar_guided's CONTROL FLOW is wired correctly --
# not that guided_roi.py's margin/AABB math actually integrates with the
# real lidar_detector.py numerics on real data. This test closes that gap,
# the same way test_lidar_detector_auto_finds_board_among_decoy_planes
# (test_camera_lidar_core.py) does for plain AUTO ROI. Only
# detect_camera_target is stubbed -- rendering a geometrically-correct
# synthetic ArUco photograph is a separately-documented test-fixture gap
# (see test_camera_lidar_core.py's module docstring), unrelated to GUIDED
# ROI itself.
# ---------------------------------------------------------------------------

def test_guided_roi_end_to_end_real_lidar_detection_crops_decoy_plane(monkeypatch):
    from tests.test_camera_lidar_core import _synthetic_board_points_lidar_frame

    target = TargetConfig()
    rng = np.random.default_rng(7)

    # Arbitrary placement of the physical board in the LiDAR frame, and the
    # real synthetic point cloud for it (board plate + circle-hole rings).
    R_L = rpy_to_rotation_matrix(8, -15, 40, degrees=True)
    t_L = np.array([2.0, 0.3, 0.5])
    board_lidar, circle_centers_board = _synthetic_board_points_lidar_frame(target, R_L, t_L, rng, n_plate=3000)
    lidar_centers_true = (R_L @ circle_centers_board.T).T + t_L  # (4,3), CORNER_ORDER, true LiDAR-frame centers

    # Large decoy floor plane, far enough from the board that a correct
    # GUIDED ROI must exclude it entirely.
    n_floor = 8000
    floor_pts = np.column_stack([
        rng.uniform(-5, 5, n_floor), rng.uniform(-5, 5, n_floor),
        np.full(n_floor, -1.0) + rng.normal(0, 0.005, n_floor),
    ])
    cloud = PointCloudFrame(
        timestamp=0.0, points=np.concatenate([floor_pts, board_lidar], axis=0), frame_id="lidar",
    )

    # The TRUE camera<->lidar extrinsic the solver must recover -- unrelated
    # to (and not derivable from) R_L/t_L above, which only places the board
    # in the synthetic cloud.
    R_CL_true = rpy_to_rotation_matrix(5, 10, -30, degrees=True)   # R_camera_from_lidar
    t_CL_true = np.array([0.1, -0.05, 0.02])
    camera_centers_true = (R_CL_true @ lidar_centers_true.T).T + t_CL_true
    T_lidar_from_camera_true = invert_transform(to_homogeneous(R_CL_true, t_CL_true))

    # A deliberately IMPERFECT Targetless prior (coarse rotation + translation
    # error, well within GuidedROIConfig's default uncertainty budget) --
    # this is the realistic case: direct_visual_lidar_calibration's estimate
    # is coarse, not exact.
    R_err = rpy_to_rotation_matrix(2.0, -1.0, 1.5, degrees=True)
    t_err = np.array([0.05, -0.03, 0.02])
    noisy_prior_T = to_homogeneous(R_err, t_err) @ T_lidar_from_camera_true

    camera_result = CameraDetectionResult(
        success=True, circle_centers=camera_centers_true, detected_ids=frozenset(CORNER_ORDER), markers_detected=4,
    )
    monkeypatch.setattr(pipeline, "detect_camera_target", lambda *a, **k: camera_result)

    guided_config = GuidedROIConfig(
        prior=TargetlessPrior(T_lidar_from_camera=noisy_prior_T, source_path="calib.json", source_key="T_lidar_camera"),
    )
    image = ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    scene = CalibrationScene(image=image, cloud=cloud, intrinsics=None, target=target, guided_roi=guided_config)

    result = calibrate_single_scene(scene, roi_mode="guided")

    assert result.success, result.failure_reason
    diagnostics = result.guided_roi_diagnostics
    assert diagnostics is not None
    assert diagnostics.fallback_to_auto_used is False  # a guided attempt found it -- no need for full-cloud AUTO
    # Real cropping happened: far fewer points reached the detector than the
    # whole cloud (floor_pts + board_lidar).
    assert result.lidar_detection.roi_point_count < floor_pts.shape[0] + board_lidar.shape[0]
    # The selected ROI actually excludes the decoy floor (z=-1.0).
    assert diagnostics.selected_roi is not None
    assert diagnostics.selected_roi.z_min > -1.0
    # Final extrinsic recovered correctly -- proving the imperfect prior
    # (only used for ROI search) did not degrade the actual solve.
    assert np.allclose(result.R_camera_from_lidar, R_CL_true, atol=0.05)
    assert np.allclose(result.t_camera_from_lidar, t_CL_true, atol=0.02)
    assert result.scene_type == SceneType.VALID_FULL


# ---------------------------------------------------------------------------
# roi_mode validation / GUIDED_ROI_PRIOR_MISSING
# ---------------------------------------------------------------------------

def test_invalid_roi_mode_raises():
    target = TargetConfig()
    image = ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    cloud = PointCloudFrame(timestamp=0.0, points=np.zeros((10, 3)))
    scene = CalibrationScene(image=image, cloud=cloud, intrinsics=None, target=target)
    with pytest.raises(ValueError):
        calibrate_single_scene(scene, roi_mode="not_a_real_mode")


def test_guided_without_config_fails_with_specific_reason(monkeypatch):
    scene = _make_scene(guided_config=None)
    monkeypatch.setattr(pipeline, "detect_camera_target", lambda *a, **k: _fake_camera_result())

    result = calibrate_single_scene(scene, roi_mode="guided")

    assert result.success is False
    assert result.failure_reason == FailureReason.GUIDED_ROI_PRIOR_MISSING
