from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import yaml

from calibration.calibration_io import StandardCalibration
from calibration.stereo import (
    StereoPairObservation,
    apply_pair_pose_diversity_scores_from_intrinsics,
    baseline_from_t,
    calibrate_stereo,
    euler_zyx_from_rotation,
    epipolar_errors,
    extract_timestamp_from_filename,
    extract_timestamp_from_sidecar,
    inverse_rt,
    match_common_charuco_corners,
    pair_image_paths,
    reject_pairs_by_id,
    rectification_vertical_errors,
    sampson_distances,
    score_stereo_pair_quality,
    stereo_pair_quality_components,
    set_pair_used,
    stats_from_values,
    transformation_from_rt,
)
from calibration.project_io import project_from_dict, project_to_dict
from calibration.stereo_auditor import (
    build_stereo_evidence_report,
    compute_capture_coach,
    compute_sync_guard,
)
from calibration.stereo_session import StereoSession
from calibration.types import CameraConfig, CameraModelType, CalibrationProject, DetectionResult, PatternConfig, PatternType
from export.stereo import (
    export_stereo_html,
    export_stereo_json,
    export_stereo_kalibr_camchain,
    export_stereo_yaml,
    StereoRoboticsExportOptions,
    stereo_pair_from_dict,
    stereo_pair_to_dict,
    stereo_pairs_from_dict,
    stereo_pairs_to_dict,
    stereo_result_from_dict,
    stereo_result_to_dict,
)
from calibration.stereo import StereoCalibrationResult


def _detection(ids, offset_x=0.0, *, board_area_ratio=None, corner_confidence=None, board_center_px=None):
    ids_arr = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    corners = np.array([[[float(i + offset_x), float(i * 2)]] for i in ids], dtype=np.float32)
    obj = np.array([[[float(i), 0.0, 0.0]] for i in ids], dtype=np.float32)
    return DetectionResult(
        image_id="d",
        success=True,
        corners=corners,
        object_points=obj,
        ids=ids_arr,
        num_corners=len(ids),
        board_area_ratio=board_area_ratio,
        corner_confidence=corner_confidence,
        board_center_px=board_center_px,
    )


def test_common_charuco_id_matching_uses_only_shared_ids():
    pair = match_common_charuco_corners(_detection([1, 2, 3, 5]), _detection([2, 3, 4, 5], 10))
    assert pair.common_ids.tolist() == [2, 3, 5]
    assert pair.common_count == 3
    assert pair.image_points_cam1.shape[0] == 3
    assert pair.image_points_cam2.shape[0] == 3


def test_transformation_inverse_baseline_and_euler():
    angle = np.deg2rad(10.0)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([[0.3], [0.0], [0.0]])
    T = transformation_from_rt(R, t)
    _R_inv, _t_inv, T_inv = inverse_rt(R, t)
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-9)
    assert baseline_from_t(t) == pytest.approx(0.3)
    roll, pitch, yaw = euler_zyx_from_rotation(R)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(10.0)


def test_epipolar_and_sampson_distances_are_near_zero_for_perfect_pairs():
    F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    pts1 = np.array([[10, 5], [20, 7], [30, 9]], dtype=np.float64)
    pts2 = np.array([[12, 5], [24, 7], [36, 9]], dtype=np.float64)
    assert np.max(epipolar_errors(F, pts1, pts2)) < 1e-9
    assert np.max(sampson_distances(F, pts1, pts2)) < 1e-9


def test_pair_reject_include_and_stats():
    pair = match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]))
    set_pair_used(pair, False, "bad sync")
    assert not pair.used
    assert pair.rejected_reason == "bad sync"
    set_pair_used(pair, True)
    assert pair.used
    stats = stats_from_values([1, 2, 3])
    assert stats.mean == pytest.approx(2.0)
    assert stats.rmse == pytest.approx(np.sqrt((1 + 4 + 9) / 3))


def test_reject_pairs_by_outlier_ids_marks_pairs_without_touching_others():
    pairs = [
        match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]), pair_id="Pair 001"),
        match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]), pair_id="Pair 002"),
    ]
    changed = reject_pairs_by_id(pairs, {"Pair 002"}, reason="auto outlier")

    assert changed == 1
    assert pairs[0].used
    assert not pairs[1].used
    assert pairs[1].rejected_reason == "auto outlier"


def test_stereo_session_filters_and_rejects_outlier_pairs():
    pair1 = match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]), pair_id="Pair 001")
    pair2 = match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]), pair_id="Pair 002")
    result = _sample_stereo_result()
    from calibration.stereo import StereoPairValidation
    result.pair_validations = [
        StereoPairValidation("Pair 001", 6, status="Good"),
        StereoPairValidation("Pair 002", 6, status="Outlier"),
    ]
    session = StereoSession([pair1, pair2], result)

    assert [pair.pair_id for pair in session.visible_pairs(outliers_only=True)] == ["Pair 002"]
    assert session.reject_outliers() == 1
    assert pair1.used
    assert not pair2.used


def test_pair_image_paths_by_stem_and_timestamp():
    cam1 = ["c1/left_001.png", "c1/left_002.png", "c1/left_003.png"]
    cam2 = ["c2/left_003.png", "c2/left_001.png", "c2/extra.png"]
    by_stem = pair_image_paths(cam1, cam2, mode="stem")
    assert by_stem.camera1_paths == ["c1/left_001.png", "c1/left_003.png"]
    assert by_stem.camera2_paths == ["c2/left_001.png", "c2/left_003.png"]
    assert by_stem.warnings

    assert extract_timestamp_from_filename("cam_1692600000123.png") == pytest.approx(1692600000.123)
    ts = pair_image_paths(
        ["c1/frame_1000.000.png", "c1/frame_1000.060.png"],
        ["c2/frame_1000.020.png", "c2/frame_1000.200.png"],
        mode="timestamp",
        max_timestamp_delta_ms=30.0,
    )
    assert ts.camera1_paths == ["c1/frame_1000.000.png"]
    assert ts.camera2_paths == ["c2/frame_1000.020.png"]
    assert ts.warnings


def test_pair_image_paths_by_ros_sidecar_timestamp(tmp_path):
    cam1_a = tmp_path / "cam1_a.png"
    cam1_b = tmp_path / "cam1_b.png"
    cam2_a = tmp_path / "cam2_a.png"
    cam1_a.write_bytes(b"")
    cam1_b.write_bytes(b"")
    cam2_a.write_bytes(b"")
    (tmp_path / "cam1_a.json").write_text('{"header_stamp": {"sec": 10, "nanosec": 0}}', encoding="utf-8")
    (tmp_path / "cam1_b.json").write_text('{"header_stamp": {"sec": 11, "nanosec": 0}}', encoding="utf-8")
    (tmp_path / "cam2_a.json").write_text('{"header_stamp": {"sec": 10, "nanosec": 20000000}}', encoding="utf-8")

    assert extract_timestamp_from_sidecar(str(cam2_a)) == pytest.approx(10.02)
    pairing = pair_image_paths(
        [str(cam1_a), str(cam1_b)],
        [str(cam2_a)],
        mode="ros_timestamp",
        max_timestamp_delta_ms=30.0,
    )

    assert pairing.camera1_paths == [str(cam1_a)]
    assert pairing.camera2_paths == [str(cam2_a)]
    assert pairing.unmatched_camera1_paths == [str(cam1_b)]


def test_stereo_pair_quality_scores_warn_for_weak_pairs():
    good_score, good_status, good_warnings = score_stereo_pair_quality(
        _detection(range(20), board_area_ratio=0.15, corner_confidence=0.95),
        _detection(range(20), board_area_ratio=0.14, corner_confidence=0.90),
        common_corners=20,
        min_common_corners=8,
        sync_delta_ms=5.0,
    )
    assert good_score >= 75
    assert good_status == "Good"
    assert not good_warnings

    bad_score, bad_status, bad_warnings = score_stereo_pair_quality(
        _detection(range(4), board_area_ratio=0.02, corner_confidence=0.4),
        _detection(range(4), board_area_ratio=0.03, corner_confidence=0.5),
        common_corners=4,
        min_common_corners=8,
        sync_delta_ms=80.0,
    )
    assert bad_score < good_score
    assert bad_status == "Reject"
    assert any("common corners" in warning for warning in bad_warnings)
    assert any("Sync delta" in warning for warning in bad_warnings)


def test_board_position_quality_uses_normalized_image_center():
    centered = stereo_pair_quality_components(
        _detection(range(12), board_center_px=(320.0, 240.0)),
        _detection(range(12), board_center_px=(320.0, 240.0)),
        common_corners=12,
        image_size_cam1=(640, 480),
        image_size_cam2=(640, 480),
    )
    corner = stereo_pair_quality_components(
        _detection(range(12), board_center_px=(0.0, 0.0)),
        _detection(range(12), board_center_px=(0.0, 0.0)),
        common_corners=12,
        image_size_cam1=(640, 480),
        image_size_cam2=(640, 480),
    )

    assert centered["board_position"] == pytest.approx(100.0)
    assert corner["board_position"] < centered["board_position"]


def test_rectification_vertical_error_zero_for_identity_rectification():
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    cal = StandardCalibration("cam", K, D, CameraModelType.PINHOLE, width=100, height=100)
    pair = match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]))
    values = rectification_vertical_errors([pair], cal, cal, np.eye(3), np.eye(3), K, K)
    assert np.max(values) < 1e-9


def _sample_stereo_result() -> StereoCalibrationResult:
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    cal1 = StandardCalibration("cam1", K, D, CameraModelType.PINHOLE, width=100, height=100)
    cal2 = StandardCalibration("cam2", K, D, CameraModelType.PINHOLE, width=100, height=100)
    return StereoCalibrationResult(
        camera1=cal1,
        camera2=cal2,
        image_size=(100, 100),
        stereo_rms=0.1,
        R_cam2_from_cam1=np.eye(3),
        t_cam2_from_cam1=np.array([[0.1], [0], [0]]),
        E=np.eye(3),
        F=np.eye(3),
        T_cam2_from_cam1=np.eye(4),
        R_cam1_from_cam2=np.eye(3),
        t_cam1_from_cam2=np.array([[-0.1], [0], [0]]),
        T_cam1_from_cam2=np.eye(4),
        baseline=0.1,
        roll_pitch_yaw_deg=(0.0, 0.0, 0.0),
    )


def test_stereo_result_yaml_json_html_export(tmp_path):
    result = _sample_stereo_result()
    result.capture_coach = {"dataset_quality_score": 82.0, "joint_coverage_score": 75.0, "dataset_ready": True}
    result.sync_guard = {"status": "GOOD", "timestamp_delta_ms": {"median": 2.0, "p95": 4.0, "max": 5.0}}
    result.calibration_audit = {
        "cross_camera_pose_consistency": {"translation_error_mm": {"p95": 1.2}},
        "reconstruction": {"point_to_pose_error_mm": {"rmse": 0.3}, "plane_error_mm": {"rmse": 0.1}},
        "stability_uncertainty": {"baseline_95ci_mm": [99.9, 100.1]},
    }
    result.evidence_report = {"confidence": "HIGH", "warnings": [], "evidence_model": "GT-free multi-evidence validation"}
    payload = stereo_result_to_dict(result)
    assert payload["stereo"]["baseline"] == pytest.approx(0.1)
    assert payload["evidence_report"]["confidence"] == "HIGH"
    json_path = export_stereo_json(result, tmp_path / "stereo.json")
    yaml_path = export_stereo_yaml(result, tmp_path / "stereo.yaml")
    html_path = export_stereo_html(result, tmp_path / "stereo.html")
    assert json.loads(open(json_path, encoding="utf-8").read())["stereo"]["baseline"] == pytest.approx(0.1)
    assert yaml.safe_load(open(yaml_path, encoding="utf-8"))["stereo"]["baseline"] == pytest.approx(0.1)
    assert "static_transform_publisher" in json.loads(open(json_path, encoding="utf-8").read())["robotics"]["static_transform_publisher_ros2_rpy_degrees"]
    html_text = open(html_path, encoding="utf-8").read()
    assert "Stereo Calibration Report" in html_text
    assert "Capture Coach" in html_text
    assert "Calibration Auditor" in html_text


def test_stereo_robotics_export_options_control_frames_and_rotation_format():
    result = _sample_stereo_result()
    payload = stereo_result_to_dict(
        result,
        StereoRoboticsExportOptions(
            parent_frame="base_left",
            child_frame="base_right",
            rotation_format="rpy_radians",
        ),
    )

    robotics = payload["robotics"]
    assert robotics["parent_frame"] == "base_left"
    assert robotics["child_frame"] == "base_right"
    assert robotics["rotation_format"] == "rpy_radians"
    assert "base_left base_right" in robotics["selected_static_transform_publisher"]
    assert len(robotics["quaternion_xyzw"]) == 4


def test_stereo_kalibr_camchain_export_contains_transform(tmp_path):
    result = _sample_stereo_result()
    path = export_stereo_kalibr_camchain(result, tmp_path / "camchain.yaml")
    payload = yaml.safe_load(open(path, encoding="utf-8"))

    assert "cam0" in payload
    assert "cam1" in payload
    assert payload["cam1"]["T_cn_cnm1"] == stereo_result_to_dict(result)["stereo"]["T_cam2_from_cam1"]
    assert payload["cam0"]["intrinsics"] == [1.0, 1.0, 0.0, 0.0]


def test_pose_diversity_from_intrinsics_updates_quality_components():
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros((5, 1), dtype=np.float64)
    cal = StandardCalibration("cam1", K, D, CameraModelType.PINHOLE, width=640, height=480)
    obj = np.array([[[x * 0.04, y * 0.04, 0.0]] for y in range(4) for x in range(5)], dtype=np.float32)
    pairs = []
    for index, yaw in enumerate([0.0, 0.25, -0.25]):
        rvec = np.array([[0.05], [yaw], [0.02]], dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [0.7 + index * 0.1]], dtype=np.float64)
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        pairs.append(
            StereoPairObservation(
                pair_id=f"Pair {index}",
                object_points=obj,
                image_points_cam1=pts.astype(np.float32),
                image_points_cam2=pts.astype(np.float32),
                common_ids=np.arange(obj.shape[0], dtype=np.int32),
                quality_components={"pose_diversity": 0.0},
            )
        )

    apply_pair_pose_diversity_scores_from_intrinsics(pairs, cal)

    assert max(pair.quality_components["pose_diversity"] for pair in pairs) > 90.0


def test_stereo_result_dict_round_trip_and_project_payload():
    result = _sample_stereo_result()
    payload = stereo_result_to_dict(result)
    restored = stereo_result_from_dict(payload)
    assert restored.baseline == pytest.approx(result.baseline)
    np.testing.assert_allclose(restored.R_cam2_from_cam1, result.R_cam2_from_cam1)
    np.testing.assert_allclose(restored.t_cam2_from_cam1, result.t_cam2_from_cam1)
    assert restored.evidence_report == result.evidence_report

    pair = StereoPairObservation(
        pair_id="Pair 007",
        object_points=np.zeros((6, 1, 3), dtype=np.float32),
        image_points_cam1=np.ones((6, 1, 2), dtype=np.float32),
        image_points_cam2=np.full((6, 1, 2), 2, dtype=np.float32),
        common_ids=np.arange(6, dtype=np.int32),
        detected_points_cam1=np.arange(16, dtype=np.float32).reshape(8, 1, 2),
        detected_points_cam2=np.arange(16, 32, dtype=np.float32).reshape(8, 1, 2),
        detected_ids_cam1=np.arange(8, dtype=np.int32),
        detected_ids_cam2=np.arange(2, 10, dtype=np.int32),
        image_path_cam1="C:/data/cam1/pair_007.png",
        image_path_cam2="C:/data/cam2/pair_007.png",
        sync_delta_ms=12.5,
        used=False,
        rejected_reason="manual reject",
        quality_score=42.0,
        quality_status="Warning",
        quality_components={"common_corners": 50.0, "pose_diversity": 20.0},
        quality_warnings=["small board"],
    )
    restored_pair = stereo_pair_from_dict(stereo_pair_to_dict(pair))
    assert restored_pair.pair_id == pair.pair_id
    assert not restored_pair.used
    assert restored_pair.rejected_reason == "manual reject"
    assert restored_pair.image_path_cam1 == pair.image_path_cam1
    np.testing.assert_allclose(restored_pair.image_points_cam2, pair.image_points_cam2)
    np.testing.assert_allclose(restored_pair.detected_points_cam1, pair.detected_points_cam1)
    assert restored_pair.detected_ids_cam2.tolist() == list(range(2, 10))
    assert restored_pair.quality_components["pose_diversity"] == pytest.approx(20.0)

    project = CalibrationProject(
        project_name="stereo project",
        camera_config=CameraConfig(width=100, height=100),
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=5, squares_y=4, square_size=0.04),
        stereo_result=payload,
        stereo_pairs=stereo_pairs_to_dict([pair]),
        stereo_export_paths={"html": "Output/stereo_report.html"},
    )
    restored_project = project_from_dict(project_to_dict(project))
    assert restored_project.stereo_result["stereo"]["baseline"] == pytest.approx(0.1)
    restored_pairs = stereo_pairs_from_dict(restored_project.stereo_pairs)
    assert len(restored_pairs) == 1
    assert restored_pairs[0].quality_status == "Warning"
    assert restored_pairs[0].common_ids.tolist() == list(range(6))
    assert restored_project.stereo_export_paths["html"] == "Output/stereo_report.html"


def test_synthetic_stereo_calibration_recovers_known_translation():
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros((5, 1), dtype=np.float64)
    cam1 = StandardCalibration("cam1", K, D, CameraModelType.PINHOLE, width=640, height=480)
    cam2 = StandardCalibration("cam2", K.copy(), D.copy(), CameraModelType.PINHOLE, width=640, height=480)
    R_cam2_from_cam1 = np.eye(3, dtype=np.float64)
    t_cam2_from_cam1 = np.array([[0.12], [0.0], [0.0]], dtype=np.float64)

    grid = np.array(
        [[[x * 0.04, y * 0.04, 0.0]] for y in range(5) for x in range(7)],
        dtype=np.float32,
    )
    pairs = []
    for i, yaw_deg in enumerate([-12, -6, 0, 7, 13, 18]):
        rvec1 = np.array([[0.05], [np.deg2rad(yaw_deg)], [0.02]], dtype=np.float64)
        R_board_cam1, _ = cv2.Rodrigues(rvec1)
        t_board_cam1 = np.array([[0.0], [0.0], [0.8 + i * 0.04]], dtype=np.float64)
        pts1, _ = cv2.projectPoints(grid, rvec1, t_board_cam1, K, D)

        R_board_cam2 = R_cam2_from_cam1 @ R_board_cam1
        t_board_cam2 = R_cam2_from_cam1 @ t_board_cam1 + t_cam2_from_cam1
        rvec2, _ = cv2.Rodrigues(R_board_cam2)
        pts2, _ = cv2.projectPoints(grid, rvec2, t_board_cam2, K, D)
        pairs.append(
            StereoPairObservation(
                pair_id=f"Pair {i:03d}",
                object_points=grid,
                image_points_cam1=pts1.astype(np.float32),
                image_points_cam2=pts2.astype(np.float32),
                common_ids=np.arange(grid.shape[0], dtype=np.int32),
            )
        )

    result = calibrate_stereo(pairs, cam1, cam2, (640, 480))
    np.testing.assert_allclose(result.R_cam2_from_cam1, R_cam2_from_cam1, atol=1e-3)
    np.testing.assert_allclose(result.t_cam2_from_cam1.reshape(3), t_cam2_from_cam1.reshape(3), atol=1e-3)
    assert result.baseline == pytest.approx(0.12, abs=1e-3)
    assert result.holdout_train_pair_count > 0
    assert result.holdout_validation_pair_count > 0
    assert result.holdout_validation_error is not None
    assert result.evidence_report["confidence"] in {"HIGH", "MEDIUM", "LOW"}
    assert "dataset_quality_score" in result.capture_coach
    assert "cross_camera_pose_consistency" in result.calibration_audit


def test_stereo_auditor_builds_capture_sync_and_evidence_report():
    pair1 = match_common_charuco_corners(
        _detection(range(12), board_area_ratio=0.12, corner_confidence=0.9, board_center_px=(320, 240)),
        _detection(range(12), board_area_ratio=0.11, corner_confidence=0.85, board_center_px=(300, 220)),
        pair_id="Pair 001",
    )
    pair2 = match_common_charuco_corners(
        _detection(range(12), board_area_ratio=0.05, corner_confidence=0.7, board_center_px=(40, 40)),
        _detection(range(12), board_area_ratio=0.05, corner_confidence=0.7, board_center_px=(50, 50)),
        pair_id="Pair 002",
    )
    pair1.sync_delta_ms = 5.0
    pair2.sync_delta_ms = 45.0
    pair1.quality_score = 80.0
    pair2.quality_score = 45.0
    pair1.quality_components = {"pose_diversity": 80.0, "timestamp_sync": 95.0}
    pair2.quality_components = {"pose_diversity": 30.0, "timestamp_sync": 20.0}

    capture = compute_capture_coach([pair1, pair2], (640, 480), target_pairs=2)
    sync = compute_sync_guard([pair1, pair2], threshold_ms=30.0)
    evidence = build_stereo_evidence_report(_sample_stereo_result(), [pair1, pair2], (640, 480))

    assert capture["usable_pairs"] == 2
    assert capture["dataset_quality_score"] > 0.0
    assert sync["status"] == "SYNC SUSPECT"
    assert sync["suspect_pair_ids"] == ["Pair 002"]
    assert evidence["evidence_report"]["absolute_accuracy_claim"] == "Not available without external ground truth."


def test_fisheye_stereo_path_dispatches_to_opencv_fisheye(monkeypatch):
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    cam = StandardCalibration("fish", K, D, CameraModelType.FISHEYE, width=100, height=100)
    pair = match_common_charuco_corners(_detection([1, 2, 3, 4, 5, 6]), _detection([1, 2, 3, 4, 5, 6]))
    calls = {"calibrate": 0, "rectify": 0}

    def fake_stereo_calibrate(object_points, image_points1, image_points2, K1, D1, K2, D2, image_size, **kwargs):
        calls["calibrate"] += 1
        return 0.2, K1, D1, K2, D2, np.eye(3), np.array([[0.2], [0.0], [0.0]])

    def fake_stereo_rectify(K1, D1, K2, D2, image_size, R, T, **kwargs):
        calls["rectify"] += 1
        return np.eye(3), np.eye(3), np.eye(3, 4), np.eye(3, 4), np.eye(4)

    monkeypatch.setattr(cv2.fisheye, "stereoCalibrate", fake_stereo_calibrate)
    monkeypatch.setattr(cv2.fisheye, "stereoRectify", fake_stereo_rectify)

    result = calibrate_stereo([pair, pair], cam, cam, (100, 100), compute_holdout=False)

    assert calls == {"calibrate": 1, "rectify": 1}
    assert result.baseline == pytest.approx(0.2)
    np.testing.assert_allclose(result.t_cam2_from_cam1.reshape(3), [0.2, 0.0, 0.0])


def test_synthetic_fisheye_stereo_calibration_recovers_known_translation():
    K = np.array([[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([[-0.02], [0.003], [0.0], [0.0]], dtype=np.float64)
    cam1 = StandardCalibration("fish1", K, D, CameraModelType.FISHEYE, width=320, height=240)
    cam2 = StandardCalibration("fish2", K.copy(), D.copy(), CameraModelType.FISHEYE, width=320, height=240)
    R_cam2_from_cam1 = np.eye(3, dtype=np.float64)
    t_cam2_from_cam1 = np.array([[0.08], [0.0], [0.0]], dtype=np.float64)

    grid = np.array(
        [[[x * 0.035, y * 0.035, 0.0]] for y in range(5) for x in range(7)],
        dtype=np.float64,
    )
    pairs = []
    for i, yaw_deg in enumerate([-10, -5, 0, 6, 12, 16]):
        rvec1 = np.array([[0.02], [np.deg2rad(yaw_deg)], [0.01]], dtype=np.float64)
        R_board_cam1, _ = cv2.Rodrigues(rvec1)
        t_board_cam1 = np.array([[0.0], [0.0], [0.65 + i * 0.03]], dtype=np.float64)
        pts1, _ = cv2.fisheye.projectPoints(grid, rvec1, t_board_cam1, K, D)

        R_board_cam2 = R_cam2_from_cam1 @ R_board_cam1
        t_board_cam2 = R_cam2_from_cam1 @ t_board_cam1 + t_cam2_from_cam1
        rvec2, _ = cv2.Rodrigues(R_board_cam2)
        pts2, _ = cv2.fisheye.projectPoints(grid, rvec2, t_board_cam2, K, D)
        pairs.append(
            StereoPairObservation(
                pair_id=f"Fisheye Pair {i:03d}",
                object_points=grid.astype(np.float32),
                image_points_cam1=pts1.astype(np.float32),
                image_points_cam2=pts2.astype(np.float32),
                common_ids=np.arange(grid.shape[0], dtype=np.int32),
            )
        )

    result = calibrate_stereo(pairs, cam1, cam2, (320, 240), compute_holdout=False)

    np.testing.assert_allclose(result.R_cam2_from_cam1, R_cam2_from_cam1, atol=1e-4)
    np.testing.assert_allclose(result.t_cam2_from_cam1.reshape(3), t_cam2_from_cam1.reshape(3), atol=1e-4)
    assert result.baseline == pytest.approx(0.08, abs=1e-4)
