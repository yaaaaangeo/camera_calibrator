"""
Tests for the ROS-independent camera_lidar/ FAST-Calib core.

The load-bearing test here is test_solver_recovers_known_transform: it
builds synthetic camera/lidar circle centers from a known
T_camera_from_lidar and checks the solver recovers it, per the "Known R/T
-> synthetic centers -> solver -> estimated R/T" requirement.

camera_detector's full ArUco-image path is not covered by a rendered
synthetic image here -- constructing a geometrically-correct synthetic
marker photograph (getting the raster-to-object-point winding exactly
right) turned out to be its own small research project, independent of
whether camera_detector.py itself is correct (it consumes real
cv2.aruco.detectMarkers() corners, whose winding convention is
well-documented/standard). That's flagged as a known test-fixture gap;
this file instead covers camera_detector's real failure path (no markers
in the image) and leaves full ArUco-image coverage for a follow-up that
adds a real or carefully-rendered fixture image.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.calibration_io import StandardCalibration
from calibration.types import CameraModelType
from camera_lidar.camera_detector import (
    CameraDetectionResult,
    detect_camera_target,
    diagnose_dictionaries,
    render_marker_overlay,
)
from camera_lidar.correspondence import match_centers, match_partial_centers
from camera_lidar.lidar_detector import LidarDetectionResult, detect_lidar_target, detect_lidar_target_auto
from camera_lidar.multi_scene import calibrate_multi_scene, compare_strict_vs_flexible
from camera_lidar.pipeline import calibrate_single_scene
from camera_lidar.solver import compute_residuals, solve_rigid_transform
from camera_lidar.target_config import CORNER_ORDER, TargetConfig, load_target_config, save_target_config
from camera_lidar.types import (
    CalibrationScene,
    CapturedScene,
    FailureReason,
    ImageFrame,
    PointCloudFrame,
    ROIConfig,
    SceneType,
)
from geometry.transform import (
    invert_transform,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rpy,
    rpy_to_rotation_matrix,
    to_homogeneous,
)

# board local Y (up, toward top_left/top_right) -> lidar +Z; board local Z
# (normal) -> lidar +X; board local X (right) -> lidar +Y. Tests that rely on
# match_partial_centers' no-reference "target held upright, not mirrored"
# disambiguation (see camera_lidar/correspondence.py) need this kind of
# physically realistic placement -- raw `board` coordinates used directly as
# "lidar frame" points have no genuine up/left-right signal (board is flat,
# Z=0 for every point), so the disambiguation's outcome would be an
# unrepresentative coincidence of iteration order rather than a real test.
_BOARD_TO_LIDAR_ALIGNMENT = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])


def _place_board_in_lidar_frame(board_points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (_BOARD_TO_LIDAR_ALIGNMENT @ board_points.T).T + offset


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

def test_solver_recovers_known_transform():
    target = TargetConfig()
    lidar_centers = target.circle_centers_board_frame() + np.array([2.0, 0.3, -0.1])

    R_true = rpy_to_rotation_matrix(5, -3, 90, degrees=True)
    t_true = np.array([0.1, -0.2, 0.05])
    camera_centers = (R_true @ lidar_centers.T).T + t_true

    R_est, t_est = solve_rigid_transform(lidar_centers, camera_centers)

    assert np.allclose(R_est, R_true, atol=1e-9)
    assert np.allclose(t_est, t_true, atol=1e-9)

    residuals = compute_residuals(lidar_centers, camera_centers, R_est, t_est)
    assert residuals.rmse < 1e-9


def test_solver_residuals_increase_with_noise():
    target = TargetConfig()
    lidar_centers = target.circle_centers_board_frame() + np.array([2.0, 0.3, -0.1])
    R_true = rpy_to_rotation_matrix(10, 5, -20, degrees=True)
    t_true = np.array([0.5, 0.1, 0.2])
    camera_centers = (R_true @ lidar_centers.T).T + t_true

    rng = np.random.default_rng(1)
    noisy = camera_centers + rng.normal(0, 0.01, camera_centers.shape)
    R_est, t_est = solve_rigid_transform(lidar_centers, noisy)
    residuals = compute_residuals(lidar_centers, noisy, R_est, t_est)

    clean_residuals = compute_residuals(
        lidar_centers, camera_centers, *solve_rigid_transform(lidar_centers, camera_centers)
    )
    assert residuals.rmse > clean_residuals.rmse


def test_solver_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        solve_rigid_transform(np.zeros((3, 3)), np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# geometry.transform (SE(3) round trips used throughout camera_lidar)
# ---------------------------------------------------------------------------

def test_transform_inverse_roundtrip():
    R = rpy_to_rotation_matrix(12, -7, 33, degrees=True)
    t = np.array([1.0, -2.0, 0.5])
    T = to_homogeneous(R, t)
    T_inv = invert_transform(T)
    assert np.allclose(T_inv @ T, np.eye(4), atol=1e-9)
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-9)


def test_rpy_and_quaternion_roundtrip():
    R = rpy_to_rotation_matrix(37, -55, 142, degrees=True)
    roll, pitch, yaw = rotation_matrix_to_rpy(R, degrees=True)
    R_from_rpy = rpy_to_rotation_matrix(roll, pitch, yaw, degrees=True)
    assert np.allclose(R_from_rpy, R, atol=1e-9)

    from geometry.transform import quaternion_to_rotation_matrix
    q = rotation_matrix_to_quaternion(R)
    R_from_q = quaternion_to_rotation_matrix(*q)
    assert np.allclose(R_from_q, R, atol=1e-9)


# ---------------------------------------------------------------------------
# correspondence
# ---------------------------------------------------------------------------

def test_correspondence_resolves_arbitrary_permutation():
    target = TargetConfig()
    lidar_centers_true_order = target.circle_centers_board_frame() + np.array([2.0, 0.3, -0.1])
    R_true = rpy_to_rotation_matrix(17, -34, 128, degrees=True)
    t_true = np.array([0.5, -0.3, 1.2])
    camera_centers = (R_true @ lidar_centers_true_order.T).T + t_true

    base = list(range(4))
    for reverse in (False, True):
        seq = base[::-1] if reverse else base
        for start in range(4):
            rotated = seq[start:] + seq[:start]
            shuffled = lidar_centers_true_order[rotated]
            result = match_centers(camera_centers, shuffled)
            assert result.residual_rmse_m < 1e-9


# ---------------------------------------------------------------------------
# lidar_detector (synthetic plane + 4-circle-hole board)
# ---------------------------------------------------------------------------

def _synthetic_board_points_lidar_frame(target: TargetConfig, R, t, rng, n_plate=3000, noise_m=0.002):
    plate_hw = target.delta_width_circles / 2 + target.circle_radius + 0.1
    plate_hh = target.delta_height_circles / 2 + target.circle_radius + 0.1

    xs = rng.uniform(-plate_hw, plate_hw, n_plate)
    ys = rng.uniform(-plate_hh, plate_hh, n_plate)
    plate_pts = np.column_stack([xs, ys, np.zeros(n_plate)])

    circle_centers_board = target.circle_centers_board_frame()

    def inside_any_circle(pts):
        mask = np.zeros(len(pts), dtype=bool)
        for c in circle_centers_board:
            d = np.linalg.norm(pts[:, :2] - c[:2], axis=1)
            mask |= d < target.circle_radius
        return mask

    plate_pts = plate_pts[~inside_any_circle(plate_pts)]

    ring_pts = []
    for c in circle_centers_board:
        theta = rng.uniform(0, 2 * np.pi, 200)
        r = target.circle_radius + rng.normal(0, 0.0015, 200)
        ring_pts.append(np.column_stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), np.zeros(200)]))
    ring_pts = np.concatenate(ring_pts, axis=0)

    board_local = np.concatenate([plate_pts, ring_pts], axis=0)
    board_local[:, 2] += rng.normal(0, noise_m, board_local.shape[0])

    lidar_pts = (R @ board_local.T).T + t
    return lidar_pts, circle_centers_board


def test_lidar_detector_recovers_circle_centers():
    rng = np.random.default_rng(0)
    target = TargetConfig()
    R = rpy_to_rotation_matrix(8, -15, 40, degrees=True)
    t = np.array([2.0, 0.3, 0.1])
    lidar_pts, circle_centers_board = _synthetic_board_points_lidar_frame(target, R, t, rng)

    cloud = PointCloudFrame(timestamp=0.0, points=lidar_pts, frame_id="lidar")
    roi = ROIConfig(x_min=0.0, x_max=5.0, y_min=-3.0, y_max=3.0, z_min=-3.0, z_max=3.0)

    result = detect_lidar_target(cloud, roi, target)
    assert result.success, result.failure_reason
    assert result.valid_circle_count == 4

    expected = (R @ circle_centers_board.T).T + t
    correspondence = match_centers(expected, result.circle_centers)
    assert correspondence.residual_rmse_m < 0.01  # < 1cm on noisy synthetic data


def test_lidar_detector_insufficient_roi_points():
    target = TargetConfig()
    cloud = PointCloudFrame(timestamp=0.0, points=np.zeros((5, 3)), frame_id="lidar")
    result = detect_lidar_target(cloud, ROIConfig(), target)
    assert not result.success
    assert result.failure_reason == FailureReason.INSUFFICIENT_ROI_POINTS


def test_lidar_detector_auto_finds_board_among_decoy_planes():
    """AUTO ROI (no manual box) must pick the target-sized board plane, not
    a much larger decoy floor/wall plane -- "a RANSAC plane was found" !=
    "the target was found" (requirement #35 in the multi-scene spec)."""
    rng = np.random.default_rng(0)
    target = TargetConfig()
    R = rpy_to_rotation_matrix(8, -15, 40, degrees=True)
    t = np.array([2.0, 0.3, 0.5])
    board_lidar, circle_centers_board = _synthetic_board_points_lidar_frame(target, R, t, rng, n_plate=3000)

    # Decoy floor: much larger extent and point count than the board.
    n_floor = 8000
    floor_pts = np.column_stack([
        rng.uniform(-5, 5, n_floor), rng.uniform(-5, 5, n_floor),
        np.full(n_floor, -1.0) + rng.normal(0, 0.005, n_floor),
    ])
    # Decoy wall.
    n_wall = 4000
    wall_pts = np.column_stack([
        np.full(n_wall, 4.0) + rng.normal(0, 0.005, n_wall),
        rng.uniform(-5, 5, n_wall), rng.uniform(-2, 3, n_wall),
    ])

    cloud = PointCloudFrame(
        timestamp=0.0,
        points=np.concatenate([floor_pts, wall_pts, board_lidar], axis=0),
        frame_id="lidar",
    )

    result = detect_lidar_target_auto(cloud, target, max_planes=6)
    assert result.success, result.failure_reason
    assert result.plane_candidate_count >= 3  # floor, wall, board at minimum

    expected = (R @ circle_centers_board.T).T + t
    correspondence = match_centers(expected, result.circle_centers)
    assert correspondence.residual_rmse_m < 0.01


def test_lidar_detector_no_plane_in_random_noise():
    rng = np.random.default_rng(2)
    target = TargetConfig()
    # Uniform 3D noise has no dominant plane -- should fail plane detection
    # (or occasionally circle/geometry stages), never crash or silently succeed.
    points = rng.uniform(-2, 2, (500, 3))
    cloud = PointCloudFrame(timestamp=0.0, points=points, frame_id="lidar")
    roi = ROIConfig(x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=-2, z_max=2)
    result = detect_lidar_target(cloud, roi, target)
    assert not result.success
    assert result.failure_reason is not None


# ---------------------------------------------------------------------------
# camera_detector (failure path -- see module docstring re: full-image gap)
# ---------------------------------------------------------------------------

def test_camera_detector_no_markers_found():
    target = TargetConfig()
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    intrinsics = StandardCalibration(
        label="synthetic",
        camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        distortion=np.zeros(5),
        model_name=CameraModelType.PINHOLE,
        width=640, height=480,
    )
    result = detect_camera_target(blank, intrinsics, target)
    assert not result.success
    assert result.failure_reason == FailureReason.CAMERA_MARKER_NOT_FOUND
    # No raw ArUco at all -- diagnostic fields still populated on this early-failure branch.
    assert result.dictionary_name == target.aruco_dictionary
    assert result.detected_marker_ids == []
    assert result.missing_marker_ids == sorted(target.marker_ids.values())


def _render_single_marker_image(marker_id: int, dictionary_name: str = "DICT_4X4_50") -> np.ndarray:
    """A REAL rendered ArUco marker image (not a synthetic/fake result) --
    fills part of the module-docstring's "known gap" for the raw-detection
    path specifically (cv2.aruco.detectMarkers()'s own decode, independent
    of the PnP object-point-winding concern that gap is actually about).
    One real marker on a plain background, comfortably below the 3-marker
    PnP threshold -- exercises the "found_ids < _MIN_MARKERS_FOR_PNP" early-
    failure branch with a REAL raw detection, not zero markers."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    canvas = np.full((400, 400), 255, dtype=np.uint8)
    canvas[100:300, 100:300] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _synthetic_intrinsics() -> StandardCalibration:
    return StandardCalibration(
        label="synthetic",
        camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        distortion=np.zeros(5),
        model_name=CameraModelType.PINHOLE,
        width=640, height=480,
    )


def test_camera_detector_single_real_marker_below_pnp_threshold():
    target = TargetConfig()  # default marker_ids: top_left=1, top_right=2, bottom_right=3, bottom_left=4
    image = _render_single_marker_image(marker_id=1)
    result = detect_camera_target(image, _synthetic_intrinsics(), target)

    assert not result.success  # 1 marker < _MIN_MARKERS_FOR_PNP
    assert result.failure_reason == FailureReason.CAMERA_MARKER_NOT_FOUND
    assert result.detected_marker_ids == [1]
    assert result.matched_marker_ids == [1]
    assert result.missing_marker_ids == [2, 3, 4]
    assert result.dictionary_name == "DICT_4X4_50"
    assert result.rejected_candidate_count >= 1  # the outer plain-background quad also gets picked up as a rejected candidate
    assert result.detected_corners_image is not None and len(result.detected_corners_image) == 1


def test_render_marker_overlay_draws_on_a_real_detection():
    target = TargetConfig()
    image = _render_single_marker_image(marker_id=1)
    result = detect_camera_target(image, _synthetic_intrinsics(), target)
    overlay = render_marker_overlay(image, result)
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)  # drawDetectedMarkers actually drew something


def test_diagnose_dictionaries_identifies_the_correct_dictionary():
    image = _render_single_marker_image(marker_id=1, dictionary_name="DICT_5X5_50")
    results = diagnose_dictionaries(image)
    assert results[0][0] == "DICT_5X5_50"
    assert results[0][1] >= 1


def test_camera_detector_circle_offset_math():
    """Direct check of the board-pose -> circle-center offset math (the
    piece camera_detector.py applies once a board pose is solved), per the
    identity-pose check called for when a full rendered fixture image
    isn't available."""
    target = TargetConfig()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 3.0])
    expected = (R @ target.circle_centers_board_frame().T).T + t
    assert np.allclose(expected, target.circle_centers_board_frame() + t)


# ---------------------------------------------------------------------------
# pipeline (failure-reason short-circuiting)
# ---------------------------------------------------------------------------

def test_pipeline_short_circuits_on_lidar_failure():
    target = TargetConfig()
    intrinsics = StandardCalibration(
        label="synthetic",
        camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        distortion=np.zeros(5),
        model_name=CameraModelType.PINHOLE,
        width=640, height=480,
    )
    # Blank image AND empty cloud -- camera stage fails first.
    scene = CalibrationScene(
        image=ImageFrame(timestamp=0.0, image=np.full((480, 640, 3), 255, dtype=np.uint8)),
        cloud=PointCloudFrame(timestamp=0.0, points=np.zeros((0, 3))),
        intrinsics=intrinsics,
        target=target,
    )
    result = calibrate_single_scene(scene)
    assert not result.success
    assert result.failure_reason == FailureReason.CAMERA_MARKER_NOT_FOUND
    assert "marker" in result.error_message.lower()


# ---------------------------------------------------------------------------
# target_config
# ---------------------------------------------------------------------------

def test_target_config_defaults():
    target = TargetConfig()
    centers = target.circle_centers_board_frame()
    assert centers.shape == (4, 3)
    marker_centers = target.marker_centers_board_frame()
    assert set(marker_centers.keys()) == {1, 2, 3, 4}


def test_target_config_save_load_roundtrip(tmp_path):
    target = TargetConfig(circle_radius=0.15, delta_width_circles=0.6)
    path = tmp_path / "target.yaml"
    save_target_config(target, str(path))
    loaded = load_target_config(str(path))
    assert loaded.circle_radius == pytest.approx(0.15)
    assert loaded.delta_width_circles == pytest.approx(0.6)
    assert loaded.marker_size == pytest.approx(target.marker_size)


# ---------------------------------------------------------------------------
# multi_scene (joint solve over pooled correspondences + outlier flagging)
# ---------------------------------------------------------------------------

def _fake_captured_scene(scene_id: str, lidar_centers: np.ndarray, camera_centers: np.ndarray, target: TargetConfig) -> CapturedScene:
    """A CapturedScene whose .scene carries a pre-baked correspondence pair
    in .metadata, for use with the monkeypatched calibrate_single_scene
    below -- avoids needing a real image/point-cloud fixture just to
    exercise the multi-scene pooling/outlier-flagging math."""
    scene = CalibrationScene(
        image=None, cloud=None, intrinsics=None, target=target,
        metadata={"pair": (lidar_centers, camera_centers)},
    )
    return CapturedScene(scene_id=scene_id, scene=scene, roi_mode="manual")


def _install_fake_single_scene_detector(monkeypatch):
    """Redirects camera_lidar.multi_scene's calibrate_single_scene to read
    the (lidar_centers, camera_centers) pair straight out of
    scene.metadata['pair'] instead of running real detection -- multi_scene
    only needs to be tested against its OWN pooling/outlier math, not against
    a whole detection pipeline (that's covered by test_pipeline_* above)."""
    import camera_lidar.multi_scene as multi_scene_module

    class _FakeResult:
        def __init__(self, lidar_centers, camera_centers):
            self.success = True
            self.failure_reason = None
            self.lidar_centers = lidar_centers
            self.camera_centers = camera_centers
            self.scene_type = SceneType.VALID_FULL

    def _fake_calibrate_single_scene(scene, roi_mode="manual", reference_transform=None):
        lidar_c, camera_c = scene.metadata["pair"]
        return _FakeResult(lidar_c, camera_c)

    monkeypatch.setattr(multi_scene_module, "calibrate_single_scene", _fake_calibrate_single_scene)


def test_multi_scene_recovers_known_transform_and_flags_outlier(monkeypatch):
    _install_fake_single_scene_detector(monkeypatch)

    target = TargetConfig()
    R_true = rpy_to_rotation_matrix(12, -8, 33, degrees=True)
    t_true = np.array([0.3, -0.1, 1.5])
    board_centers = target.circle_centers_board_frame()
    rng = np.random.default_rng(5)

    captured = []
    for i in range(5):
        lidar_c = board_centers + rng.normal(0, 0.3, 3)
        camera_c = (R_true @ lidar_c.T).T + t_true
        if i == 4:
            camera_c = camera_c + rng.normal(0, 0.05, camera_c.shape)  # injected outlier scene
        captured.append(_fake_captured_scene(f"scene_{i}", lidar_c, camera_c, target))

    result = calibrate_multi_scene(captured)

    assert result.success
    assert result.scene_count == 5
    assert np.allclose(result.R_camera_from_lidar, R_true, atol=0.05)
    assert np.allclose(result.t_camera_from_lidar, t_true, atol=0.02)
    assert result.outlier_scene_ids == ["scene_4"]
    assert all(not s.is_outlier for s in result.per_scene if s.scene_id != "scene_4")


def test_multi_scene_excludes_scenes_marked_not_included(monkeypatch):
    _install_fake_single_scene_detector(monkeypatch)

    target = TargetConfig()
    board_centers = target.circle_centers_board_frame()

    good = _fake_captured_scene("good_1", board_centers, board_centers, target)
    good2 = _fake_captured_scene("good_2", board_centers, board_centers, target)
    excluded = _fake_captured_scene("excluded", board_centers, board_centers + 5.0, target)
    excluded.included = False

    result = calibrate_multi_scene([good, good2, excluded])
    assert result.success
    assert result.scene_count == 2
    assert "excluded" not in [s.scene_id for s in result.per_scene]


def test_multi_scene_fails_with_fewer_than_two_valid_scenes(monkeypatch):
    _install_fake_single_scene_detector(monkeypatch)

    target = TargetConfig()
    board_centers = target.circle_centers_board_frame()
    only_one = _fake_captured_scene("only", board_centers, board_centers, target)

    result = calibrate_multi_scene([only_one])
    assert not result.success
    assert result.failure_reason == FailureReason.NOT_ENOUGH_VALID_SCENES


# ---------------------------------------------------------------------------
# camera_detector.detected_ids (canonical common-feature classification)
# ---------------------------------------------------------------------------

def test_camera_detector_detected_ids_empty_on_failure():
    target = TargetConfig()
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    intrinsics = StandardCalibration(
        label="synthetic",
        camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        distortion=np.zeros(5), model_name=CameraModelType.PINHOLE, width=640, height=480,
    )
    result = detect_camera_target(blank, intrinsics, target)
    assert result.detected_ids == frozenset()


# ---------------------------------------------------------------------------
# lidar_detector partial (3-circle) detection
# ---------------------------------------------------------------------------

def test_lidar_detector_accepts_3_of_4_circles():
    """A scene missing one circle hole (LiDAR simply never cut/found it)
    must still succeed as a 3-circle PARTIAL detection, not hard-fail."""
    rng = np.random.default_rng(0)
    target = TargetConfig()
    circle_centers_board = target.circle_centers_board_frame()

    plate_hw = target.delta_width_circles / 2 + target.circle_radius + 0.1
    plate_hh = target.delta_height_circles / 2 + target.circle_radius + 0.1
    n_plate = 3000
    xs = rng.uniform(-plate_hw, plate_hw, n_plate)
    ys = rng.uniform(-plate_hh, plate_hh, n_plate)
    plate_pts = np.column_stack([xs, ys, np.zeros(n_plate)])

    def inside_any_circle(pts, centers):
        mask = np.zeros(len(pts), dtype=bool)
        for c in centers:
            mask |= np.linalg.norm(pts[:, :2] - c[:2], axis=1) < target.circle_radius
        return mask
    plate_pts = plate_pts[~inside_any_circle(plate_pts, circle_centers_board)]

    ring_pts = []
    for c in circle_centers_board[:3]:  # only cut 3 of the 4 holes
        theta = rng.uniform(0, 2 * np.pi, 200)
        r = target.circle_radius + rng.normal(0, 0.0015, 200)
        ring_pts.append(np.column_stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), np.zeros(200)]))
    ring_pts = np.concatenate(ring_pts, axis=0)

    # fill the 4th circle's area with plate material (no hole there)
    n_fill = 300
    c4 = circle_centers_board[3]
    theta = rng.uniform(0, 2 * np.pi, n_fill)
    r = np.sqrt(rng.uniform(0, 1, n_fill)) * target.circle_radius
    fill_pts = np.column_stack([c4[0] + r * np.cos(theta), c4[1] + r * np.sin(theta), np.zeros(n_fill)])

    board_local = np.concatenate([plate_pts, ring_pts, fill_pts], axis=0)
    board_local[:, 2] += rng.normal(0, 0.002, board_local.shape[0])

    R = rpy_to_rotation_matrix(8, -15, 40, degrees=True)
    t = np.array([2.0, 0.3, 0.5])
    board_lidar = (R @ board_local.T).T + t

    cloud = PointCloudFrame(timestamp=0.0, points=board_lidar, frame_id="lidar")
    roi = ROIConfig(x_min=0.0, x_max=5.0, y_min=-3.0, y_max=3.0, z_min=-3.0, z_max=3.0)
    result = detect_lidar_target(cloud, roi, target)

    assert result.success, result.failure_reason
    assert result.valid_circle_count == 3
    assert result.circle_centers.shape == (3, 3)


def test_lidar_detector_rejects_below_3_circles():
    """A totally empty/tiny cloud can't even attempt 3-circle detection."""
    target = TargetConfig()
    cloud = PointCloudFrame(timestamp=0.0, points=np.zeros((5, 3)), frame_id="lidar")
    result = detect_lidar_target(cloud, ROIConfig(), target)
    assert not result.success


# ---------------------------------------------------------------------------
# correspondence: partial matching, canonical IDs, ambiguity + reference tie-break
# ---------------------------------------------------------------------------

def test_match_partial_centers_camera_missing_one_corner():
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    R_true = rpy_to_rotation_matrix(17, -34, 128, degrees=True)
    t_true = np.array([0.5, -0.3, 1.2])
    camera_full = (R_true @ board.T).T + t_true

    camera_ids_to_centers = {
        cid: camera_full[i] for i, cid in enumerate(CORNER_ORDER) if cid != "bottom_right"
    }
    lidar_cyclic = board[[2, 3, 0, 1]]  # lidar found all 4, some rotation

    result = match_partial_centers(camera_ids_to_centers, lidar_cyclic)
    assert result is not None
    assert result.common_ids == frozenset({"top_left", "top_right", "bottom_left"})
    assert result.residual_rmse_m < 1e-6
    # LiDAR still offers 4 points (only camera is down to 3 trusted IDs), so
    # LiDAR's own "which point to drop" choice is still a real tie (resolved
    # via the upright-target fallback here, with no reference_transform given).
    assert result.ambiguous is True


def test_match_partial_centers_both_sides_missing_same_corner_is_unambiguous():
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    R_true = rpy_to_rotation_matrix(12, -8, 33, degrees=True)
    t_true = np.array([0.3, -0.1, 1.5])
    camera_full = (R_true @ board.T).T + t_true

    camera_ids_to_centers = {cid: camera_full[i] for i, cid in enumerate(CORNER_ORDER) if cid != "top_right"}
    lidar_partial = board[[2, 3, 0]]  # BR, BL, TL (also excludes top_right's point)

    result = match_partial_centers(camera_ids_to_centers, lidar_partial)
    assert result is not None
    assert result.common_ids == frozenset({"top_left", "bottom_right", "bottom_left"})
    assert result.residual_rmse_m < 1e-6
    assert result.ambiguous is False


def test_match_partial_centers_is_ambiguous_without_reference_and_resolved_with_one():
    """Hard mathematical fact: any 3-of-4 rectangle corners form the exact
    SAME triangle (sides = width, height, diagonal), so when one side (here:
    camera, with all 4 known) offers multiple congruent candidate windows,
    residual alone cannot break the tie -- verified to tie even under
    realistic detection noise. A reference_transform (e.g. from an
    already-captured FULL scene) is required to disambiguate."""
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    R_true = rpy_to_rotation_matrix(12, -8, 33, degrees=True)
    t_true = np.array([0.3, -0.1, 1.5])
    camera_full = (R_true @ board.T).T + t_true
    camera_ids_to_centers = {cid: camera_full[i] for i, cid in enumerate(CORNER_ORDER)}

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.001, (4, 3))
    lidar_partial = (board + noise)[[2, 3, 0]]  # missing top_right's point (index 1), realistic noise

    no_ref = match_partial_centers(camera_ids_to_centers, lidar_partial)
    assert no_ref.ambiguous is True

    with_ref = match_partial_centers(camera_ids_to_centers, lidar_partial, reference_transform=(R_true, t_true))
    assert with_ref.ambiguous is False
    assert with_ref.common_ids == frozenset({"top_left", "bottom_right", "bottom_left"})


def test_match_partial_centers_insufficient_points_returns_none():
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    camera_ids_to_centers = {"top_left": board[0], "top_right": board[1]}
    result = match_partial_centers(camera_ids_to_centers, board[[0, 1]])
    assert result is None


def test_match_centers_resolves_the_full_scene_klein_four_ambiguity_without_reference():
    """Regression test for a critical bug found during implementation: a
    rectangle's symmetry group has 4 elements (identity, 180-degree in-plane
    rotation, and 2 mirror reflections), and because the 4 circle centers
    are coplanar, ALL 4 of those symmetry images achieve exactly zero
    residual against ANY camera pose -- not just the 3-of-4 PARTIAL case,
    the FULL 4/4 case too. An early implementation only checked one axis
    (top/bottom) and silently returned a mirrored-wrong correspondence
    about half the time. This checks every one of the 8 raw detection
    orderings a real LiDAR detector could produce, across several
    unrelated rig rotations, with a physically realistic board placement
    (the disambiguation needs genuine up/left-right signal -- see
    _place_board_in_lidar_frame)."""
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    board_in_lidar = _place_board_in_lidar_frame(board, np.array([3.0, 0.0, 0.0]))

    base = list(range(4))
    for angles in [(20, 5, -15), (17, -34, 128), (8, -15, 40)]:
        R_true = rpy_to_rotation_matrix(*angles, degrees=True)
        t_true = np.array([0.05, 0.02, 0.01])
        camera_full = (R_true @ board_in_lidar.T).T + t_true
        for reverse in (False, True):
            seq = base[::-1] if reverse else base
            for start in range(4):
                rotated = seq[start:] + seq[:start]
                result = match_centers(camera_full, board_in_lidar[rotated])
                assert np.allclose(result.R_camera_from_lidar, R_true, atol=1e-4), (
                    f"angles={angles} rotated={rotated}: got wrong (likely 180-flipped or "
                    f"mirrored) correspondence"
                )
                assert np.allclose(result.t_camera_from_lidar, t_true, atol=1e-4)


def test_match_partial_centers_reference_transform_resolves_full_scene_tie_too():
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    board_in_lidar = _place_board_in_lidar_frame(board, np.array([3.0, 0.0, 0.0]))
    R_true = rpy_to_rotation_matrix(17, -34, 128, degrees=True)
    t_true = np.array([0.05, 0.02, 0.01])
    camera_full = (R_true @ board_in_lidar.T).T + t_true
    camera_ids_to_centers = dict(zip(CORNER_ORDER, camera_full))

    # the "twin" ordering, fed with an explicit (correct) reference transform
    twin_lidar = board_in_lidar[[2, 3, 0, 1]]
    result = match_partial_centers(camera_ids_to_centers, twin_lidar, reference_transform=(R_true, t_true))
    assert result.ambiguous is False
    assert np.allclose(result.R_camera_from_lidar, R_true, atol=1e-6)


# ---------------------------------------------------------------------------
# pipeline: FULL/PARTIAL/INVALID classification via common-feature intersection
# ---------------------------------------------------------------------------

def _fake_detection_scene(target, camera_result, lidar_result):
    scene = CalibrationScene(
        image=ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8)),
        cloud=PointCloudFrame(timestamp=0.0, points=np.zeros((1, 3))),
        intrinsics=None, target=target,
    )
    return scene, camera_result, lidar_result


def test_pipeline_classifies_full_scene(monkeypatch):
    import camera_lidar.pipeline as pl
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    board_in_lidar = _place_board_in_lidar_frame(board, np.array([3.0, 0.0, 0.0]))
    R_true = rpy_to_rotation_matrix(10, 5, -20, degrees=True)
    t_true = np.array([0.05, 0.02, 0.0])
    camera_full = (R_true @ board_in_lidar.T).T + t_true

    camera_result = CameraDetectionResult(
        success=True, circle_centers=camera_full, detected_ids=frozenset(CORNER_ORDER),
        markers_detected=4, markers_expected=4,
    )
    lidar_result = LidarDetectionResult(success=True, circle_centers=board_in_lidar, valid_circle_count=4)
    scene, _, _ = _fake_detection_scene(target, camera_result, lidar_result)

    monkeypatch.setattr(pl, "detect_camera_target", lambda *a, **k: camera_result)
    monkeypatch.setattr(pl, "detect_lidar_target", lambda *a, **k: lidar_result)

    result = pl.calibrate_single_scene(scene, roi_mode="manual")
    assert result.success
    assert result.scene_type == SceneType.VALID_FULL
    assert result.common_ids == frozenset(CORNER_ORDER)
    assert result.missing_from_camera == frozenset()
    assert result.missing_from_lidar == frozenset()
    assert np.allclose(result.R_camera_from_lidar, R_true, atol=1e-4)
    assert np.allclose(result.t_camera_from_lidar, t_true, atol=1e-4)


def test_pipeline_classifies_partial_scene_with_correct_missing_id(monkeypatch):
    import camera_lidar.pipeline as pl
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    board_in_lidar = _place_board_in_lidar_frame(board, np.array([3.0, 0.0, 0.0]))
    R_true = rpy_to_rotation_matrix(12, -8, 33, degrees=True)
    t_true = np.array([0.05, 0.02, 0.0])
    camera_full = (R_true @ board_in_lidar.T).T + t_true

    camera_result = CameraDetectionResult(
        success=True, circle_centers=camera_full, detected_ids=frozenset(CORNER_ORDER),
        markers_detected=4, markers_expected=4,
    )
    lidar_partial = board_in_lidar[[0, 1, 2]]  # TL, TR, BR (missing bottom_left)
    lidar_result = LidarDetectionResult(success=True, circle_centers=lidar_partial, valid_circle_count=3)
    scene, _, _ = _fake_detection_scene(target, camera_result, lidar_result)

    monkeypatch.setattr(pl, "detect_camera_target", lambda *a, **k: camera_result)
    monkeypatch.setattr(pl, "detect_lidar_target", lambda *a, **k: lidar_result)

    result = pl.calibrate_single_scene(scene, roi_mode="manual")
    assert result.success
    assert result.scene_type == SceneType.VALID_PARTIAL
    assert result.common_ids == frozenset({"top_left", "top_right", "bottom_right"})
    assert result.missing_from_lidar == frozenset({"bottom_left"})
    assert result.missing_from_camera == frozenset()


def test_pipeline_invalid_when_correspondence_finds_no_match(monkeypatch):
    """Direct wiring test for pipeline.py's INSUFFICIENT_COMMON_FEATURES
    path. Note: given camera_detector's own >=3-marker PnP requirement and
    lidar_detector's own >=3-circle requirement, a *real* "true common < 3"
    scenario (e.g. camera and LiDAR each independently missing a
    *different* corner) is not constructible through the real detectors --
    match_partial_centers always returns exactly `length = min(camera, lidar)`
    common ids by construction (see its module docstring's "true
    2-common-feature" scoping note), so this path is currently only
    reachable if match_partial_centers itself returns None (e.g. a future
    change relaxes one detector's floor below 3). Tested directly here so
    the branch has coverage regardless."""
    import camera_lidar.pipeline as pl
    target = TargetConfig()
    board = target.circle_centers_board_frame()

    camera_result = CameraDetectionResult(
        success=True, circle_centers=board, detected_ids=frozenset({"top_left", "top_right", "bottom_right"}),
        markers_detected=3, markers_expected=4,
    )
    lidar_result = LidarDetectionResult(success=True, circle_centers=board[:3], valid_circle_count=3)
    scene, _, _ = _fake_detection_scene(target, camera_result, lidar_result)

    monkeypatch.setattr(pl, "detect_camera_target", lambda *a, **k: camera_result)
    monkeypatch.setattr(pl, "detect_lidar_target", lambda *a, **k: lidar_result)
    monkeypatch.setattr(pl, "match_partial_centers", lambda *a, **k: None)

    result = pl.calibrate_single_scene(scene, roi_mode="manual")
    assert not result.success
    assert result.failure_reason == FailureReason.INSUFFICIENT_COMMON_FEATURES
    assert result.scene_type == SceneType.INVALID
    assert result.scene_type == SceneType.INVALID


def test_pipeline_detection_order_independence(monkeypatch):
    """Shuffling LiDAR's raw circle detection order (among the 8 valid
    cyclic rotations/directions a real detector could actually produce --
    see lidar_detector._cyclic_order) must never change the resolved
    correspondence/classification -- correspondence must never be built
    from detection order.

    Uses a physically realistic board placement (local "up" aligned with
    the LiDAR's own +Z, REP-103 convention) because disambiguating the
    FULL 4/4 case's inherent 180-degree in-plane tie (see
    correspondence.py's module docstring) falls back to the "target held
    upright" assumption when, as here, no reference_transform is available
    yet (this is the very first/only scene)."""
    import camera_lidar.pipeline as pl
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    # board local Y (up, toward top_left/top_right) -> lidar +Z; board local Z (normal) -> lidar +X.
    alignment = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    board_in_lidar = (alignment @ board.T).T + np.array([3.0, 0.0, 0.0])

    R_true = rpy_to_rotation_matrix(20, 5, -15, degrees=True)
    t_true = np.array([0.05, 0.02, 0.0])  # realistic small camera-lidar rig baseline
    camera_full = (R_true @ board_in_lidar.T).T + t_true

    camera_result = CameraDetectionResult(
        success=True, circle_centers=camera_full, detected_ids=frozenset(CORNER_ORDER),
        markers_detected=4, markers_expected=4,
    )

    base = list(range(4))
    for reverse in (False, True):
        seq = base[::-1] if reverse else base
        for start in range(4):
            rotated = seq[start:] + seq[:start]
            shuffled_lidar = board_in_lidar[rotated]
            lidar_result = LidarDetectionResult(success=True, circle_centers=shuffled_lidar, valid_circle_count=4)
            scene, _, _ = _fake_detection_scene(target, camera_result, lidar_result)

            monkeypatch.setattr(pl, "detect_camera_target", lambda *a, **k: camera_result)
            monkeypatch.setattr(pl, "detect_lidar_target", lambda *a, **k: lidar_result)

            result = pl.calibrate_single_scene(scene, roi_mode="manual")
            assert result.success
            assert result.scene_type == SceneType.VALID_FULL
            assert np.allclose(result.R_camera_from_lidar, R_true, atol=1e-4), f"failed for rotated={rotated}"
            assert np.allclose(result.t_camera_from_lidar, t_true, atol=1e-4), f"failed for rotated={rotated}"


# ---------------------------------------------------------------------------
# Calibration Policy: STRICT (4/4 only) / FLEXIBLE (>=3/4) / COMPARE BOTH
# ---------------------------------------------------------------------------

def _install_fake_classifying_detector(monkeypatch):
    """A more realistic fake than _install_fake_single_scene_detector: runs
    the REAL correspondence/classification logic (match_partial_centers)
    against per-scene fake camera/lidar detections stashed in
    scene.metadata, so STRICT/FLEXIBLE filtering has real scene_type values
    to filter on, and PARTIAL scenes exercise the real reference-transform
    disambiguation path."""
    import camera_lidar.multi_scene as multi_scene_module
    from camera_lidar.solver import compute_residuals as _compute_residuals
    from camera_lidar.types import CameraLidarCalibrationResult, FailureReason as _FailureReason
    from geometry.transform import invert_transform as _invert, to_homogeneous as _to_homog

    def _fake_calibrate_single_scene(scene, roi_mode="manual", reference_transform=None):
        cam = scene.metadata["fake_camera"]
        lidar = scene.metadata["fake_lidar"]
        camera_ids_to_centers = {
            cid: cam.circle_centers[i] for i, cid in enumerate(CORNER_ORDER) if cid in cam.detected_ids
        }
        corr = match_partial_centers(camera_ids_to_centers, lidar.circle_centers, reference_transform=reference_transform)
        if corr is None or len(corr.common_ids) < 3:
            return CameraLidarCalibrationResult(success=False, failure_reason=_FailureReason.INSUFFICIENT_COMMON_FEATURES)
        R, t = corr.R_camera_from_lidar, corr.t_camera_from_lidar
        res = _compute_residuals(corr.lidar_centers_matched, corr.camera_centers, R, t)
        T = _to_homog(R, t)
        scene_type = SceneType.VALID_FULL if len(corr.common_ids) == 4 else SceneType.VALID_PARTIAL
        return CameraLidarCalibrationResult(
            success=True, R_camera_from_lidar=R, t_camera_from_lidar=t, T_camera_from_lidar=T,
            T_lidar_from_camera=_invert(T), camera_centers=corr.camera_centers, lidar_centers=corr.lidar_centers_matched,
            residual_rmse_m=res.rmse, residual_mean_m=res.mean, residual_median_m=res.median,
            residual_p95_m=res.p95, residual_max_m=res.max,
            scene_type=scene_type, common_ids=corr.common_ids, correspondence_ambiguous=corr.ambiguous,
        )

    monkeypatch.setattr(multi_scene_module, "calibrate_single_scene", _fake_calibrate_single_scene)


def _classifying_captured_scene(scene_id, target, board, R_true, t_true, offset, missing_id, rng):
    board_in_lidar = _place_board_in_lidar_frame(board, offset)
    lidar_c = board_in_lidar + rng.normal(0, 0.0005, (4, 3))
    camera_c = (R_true @ board_in_lidar.T).T + t_true + rng.normal(0, 0.0005, (4, 3))
    detected_ids = frozenset(CORNER_ORDER) - ({missing_id} if missing_id else set())
    fake_camera = CameraDetectionResult(success=True, circle_centers=camera_c, detected_ids=detected_ids)
    if missing_id:
        idx = CORNER_ORDER.index(missing_id)
        fake_lidar = LidarDetectionResult(success=True, circle_centers=np.delete(lidar_c, idx, axis=0), valid_circle_count=3)
    else:
        fake_lidar = LidarDetectionResult(success=True, circle_centers=lidar_c, valid_circle_count=4)

    scene = CalibrationScene(
        image=ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8)),
        cloud=PointCloudFrame(timestamp=0.0, points=np.zeros((1, 3))),
        intrinsics=None, target=target,
    )
    scene.metadata["fake_camera"] = fake_camera
    scene.metadata["fake_lidar"] = fake_lidar
    return CapturedScene(scene_id=scene_id, scene=scene, roi_mode="manual")


def test_calibration_policy_strict_uses_only_full_scenes(monkeypatch):
    _install_fake_classifying_detector(monkeypatch)
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    R_true = rpy_to_rotation_matrix(15, -10, 25, degrees=True)
    t_true = np.array([0.4, 0.05, 1.2])
    rng = np.random.default_rng(1)

    scenes = [
        _classifying_captured_scene("full_1", target, board, R_true, t_true, np.array([2.0, 0.0, 0.0]), None, rng),
        _classifying_captured_scene("full_2", target, board, R_true, t_true, np.array([2.0, 0.3, 0.0]), None, rng),
        _classifying_captured_scene("full_3", target, board, R_true, t_true, np.array([2.2, -0.2, 0.1]), None, rng),
        _classifying_captured_scene("partial_1", target, board, R_true, t_true, np.array([2.5, 0.1, 0.0]), "bottom_left", rng),
        _classifying_captured_scene("partial_2", target, board, R_true, t_true, np.array([1.8, -0.3, 0.2]), "top_right", rng),
    ]

    strict_result = calibrate_multi_scene(scenes, policy="strict")
    assert strict_result.success
    assert strict_result.scene_count == 3
    assert strict_result.policy == "strict"

    flexible_result = calibrate_multi_scene(scenes, policy="flexible")
    assert flexible_result.success
    assert flexible_result.scene_count == 5
    assert flexible_result.policy == "flexible"

    # PARTIAL scenes' correspondence should be disambiguated (not ambiguous)
    # via the reference transform built from the FULL scenes.
    for c in scenes:
        if c.scene_id.startswith("partial"):
            assert c.detection.correspondence_ambiguous is False


def test_compare_strict_vs_flexible_reports_low_impact_for_consistent_partials(monkeypatch):
    _install_fake_classifying_detector(monkeypatch)
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    R_true = rpy_to_rotation_matrix(15, -10, 25, degrees=True)
    t_true = np.array([0.4, 0.05, 1.2])
    rng = np.random.default_rng(2)

    scenes = [
        _classifying_captured_scene("full_1", target, board, R_true, t_true, np.array([2.0, 0.0, 0.0]), None, rng),
        _classifying_captured_scene("full_2", target, board, R_true, t_true, np.array([2.0, 0.3, 0.0]), None, rng),
        _classifying_captured_scene("full_3", target, board, R_true, t_true, np.array([2.2, -0.2, 0.1]), None, rng),
        _classifying_captured_scene("partial_1", target, board, R_true, t_true, np.array([2.5, 0.1, 0.0]), "bottom_left", rng),
    ]

    comparison = compare_strict_vs_flexible(scenes)
    assert comparison.strict_result.success
    assert comparison.flexible_result.success
    assert comparison.impact == "LOW"
    assert comparison.translation_difference_m < 0.005
    assert comparison.rotation_difference_deg < 0.5


def test_compare_strict_vs_flexible_reports_high_impact_for_a_biased_partial(monkeypatch):
    _install_fake_classifying_detector(monkeypatch)
    target = TargetConfig()
    board = target.circle_centers_board_frame()
    R_true = rpy_to_rotation_matrix(15, -10, 25, degrees=True)
    t_true = np.array([0.4, 0.05, 1.2])
    rng = np.random.default_rng(3)

    scenes = [
        _classifying_captured_scene("full_1", target, board, R_true, t_true, np.array([2.0, 0.0, 0.0]), None, rng),
        _classifying_captured_scene("full_2", target, board, R_true, t_true, np.array([2.0, 0.3, 0.0]), None, rng),
        _classifying_captured_scene("full_3", target, board, R_true, t_true, np.array([2.2, -0.2, 0.1]), None, rng),
    ]
    # A badly-biased "partial" scene (large systematic offset, not just noise) --
    # simulate by using a visibly wrong camera pose for this one scene.
    R_bad = rpy_to_rotation_matrix(15, -10, 25 + 15, degrees=True)  # 15 deg yaw off
    offset = np.array([2.3, 0.0, 0.0])
    board_in_lidar = _place_board_in_lidar_frame(board, offset)
    lidar_c = board_in_lidar
    camera_c = (R_bad @ board_in_lidar.T).T + t_true
    idx = CORNER_ORDER.index("bottom_left")
    fake_camera = CameraDetectionResult(success=True, circle_centers=camera_c, detected_ids=frozenset(CORNER_ORDER))
    fake_lidar = LidarDetectionResult(success=True, circle_centers=np.delete(lidar_c, idx, axis=0), valid_circle_count=3)
    bad_scene = CalibrationScene(
        image=ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8)),
        cloud=PointCloudFrame(timestamp=0.0, points=np.zeros((1, 3))),
        intrinsics=None, target=target,
    )
    bad_scene.metadata["fake_camera"] = fake_camera
    bad_scene.metadata["fake_lidar"] = fake_lidar
    scenes.append(CapturedScene(scene_id="partial_bad", scene=bad_scene, roi_mode="manual"))

    comparison = compare_strict_vs_flexible(scenes)
    assert comparison.strict_result.success
    assert comparison.flexible_result.success
    assert comparison.impact == "HIGH"
