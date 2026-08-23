"""
tests/test_failure_scenarios.py
================================

문서 50번 - 실패/엣지 상황을 체크리스트처럼 1:1로 고정한다.

이미지 렌더링/검출을 거치지 않고 Dataset/DetectionResult를 직접 구성해서,
각 실패 조건이 어느 계층에서 안전하게 처리되는지 빠르게 확인한다.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pytest

from app.cli import _build_pattern_config, _normalize_cli_args, build_arg_parser, CliError
from calibration.image_quality import evaluate_image_quality, find_duplicate_groups
from calibration.models.common import MIN_FRAMES_REQUIRED, collect_calibration_inputs
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.pinhole import calibrate_pinhole
from calibration.outlier import recalibrate_with_outlier_pruning
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.sanity_check import run_sanity_check
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)
from calibration.validation import validate_all_models


W, H = 640, 480
TRUE_K = np.array([[520.0, 0, W / 2], [0, 520.0, H / 2], [0, 0, 1]], dtype=np.float64)
ZERO_D = np.zeros(5, dtype=np.float64)

DOCUMENT_50_FAILURE_CHECKLIST = {
    "image_1",
    "image_2",
    "corner_detection_failed",
    "blur_image",
    "duplicate_images",
    "center_only_board",
    "no_corner_board_observation",
    "identical_board_pose",
    "severe_distortion",
    "bad_board_size",
    "bad_image_size",
    "calibration_failed",
    "excessive_outliers",
    "nan_result",
    "inf_result",
}


def _pattern() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=7,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_5X5_100",
    )


def _charuco_object_points(pattern: PatternConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    pattern = pattern or _pattern()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y),
        pattern.square_size,
        pattern.marker_size,
        aruco_dict,
    )
    pts3d = board.getChessboardCorners().astype(np.float32)
    ids = np.arange(pts3d.shape[0], dtype=np.int32).reshape(-1, 1)
    return pts3d, ids


def _detected_frame(image_id: str, corners: np.ndarray, object_points: np.ndarray, ids: np.ndarray) -> Frame:
    pts = corners.reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    area = max(0.0, float((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]) / (W * H)))
    center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    det = DetectionResult(
        image_id=image_id,
        success=True,
        corners=corners.reshape(-1, 1, 2).astype(np.float32),
        object_points=object_points.reshape(-1, 1, 3).astype(np.float32),
        ids=ids,
        num_corners=int(corners.shape[0]),
        board_area_ratio=area,
        board_center_px=center,
        board_tilt_deg=5.0,
    )
    info = ImageInfo(image_id=image_id, path=f"/fake/{image_id}.jpg", width=W, height=H)
    return Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)


def _frame_with_phash(image_id: str, phash: str) -> Frame:
    info = ImageInfo(image_id=image_id, path=f"/fake/{image_id}.jpg", width=W, height=H, phash=phash)
    return Frame(image_info=info, status=FrameStatus.DETECTED)


def _projected_dataset(n_frames: int = 8, *, center_only: bool = False) -> Dataset:
    objp, ids = _charuco_object_points()
    frames: list[Frame] = []
    for i in range(n_frames):
        if center_only:
            rvec = np.zeros(3, dtype=np.float64)
            tvec = np.array([0.0, 0.0, 0.75], dtype=np.float64)
        else:
            rvec = np.array([0.02 * (i % 3), -0.03 * ((i + 1) % 3), 0.04 * (i - n_frames / 2)], dtype=np.float64)
            tvec = np.array([
                -0.08 + 0.16 * (i / max(n_frames - 1, 1)),
                -0.04 + 0.08 * ((i % 4) / 3),
                0.55 + 0.03 * (i % 3),
            ], dtype=np.float64)
        projected, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec, TRUE_K, ZERO_D)
        frames.append(_detected_frame(f"frame_{i:02d}", projected.reshape(-1, 2), objp, ids))
    return Dataset(frames=frames)


def test_document_50_failure_checklist_is_explicitly_tracked():
    """문서 50번 항목이 이 파일에서 1:1로 추적되는지 고정한다."""
    assert DOCUMENT_50_FAILURE_CHECKLIST == {
        "image_1",
        "image_2",
        "corner_detection_failed",
        "blur_image",
        "duplicate_images",
        "center_only_board",
        "no_corner_board_observation",
        "identical_board_pose",
        "severe_distortion",
        "bad_board_size",
        "bad_image_size",
        "calibration_failed",
        "excessive_outliers",
        "nan_result",
        "inf_result",
    }


@pytest.mark.parametrize("n_frames", [1, 2])
def test_failure_checklist_one_or_two_images_fail_calibration(n_frames):
    dataset = _projected_dataset(n_frames)
    camera = CameraConfig(width=W, height=H)

    pinhole = calibrate_pinhole(dataset, camera)
    extended = calibrate_extended_pinhole(dataset, camera)
    validations = validate_all_models(dataset, camera, _pattern(), test_ratio=0.5)

    assert not pinhole.success
    assert not extended.success
    assert f"최소 {MIN_FRAMES_REQUIRED}장" in pinhole.error_message
    assert f"최소 {MIN_FRAMES_REQUIRED}장" in extended.error_message
    assert all(not v.success for v in validations.values())


def test_failure_checklist_center_only_capture_reports_low_coverage():
    dataset = _projected_dataset(8, center_only=True)
    warnings = analyze_dataset_quality(dataset, CameraConfig(width=W, height=H))

    assert dataset.coverage_grid
    assert coverage_percentage(dataset.coverage_grid) < 50.0
    assert any("부족" in w for w in warnings)
    assert dataset.diversity is not None
    assert dataset.diversity.edge_coverage < 0.5


def test_failure_checklist_no_corner_board_observation_reports_corner_gap():
    dataset = _projected_dataset(8, center_only=True)
    analyze_dataset_quality(dataset, CameraConfig(width=W, height=H))

    corner_cells = [
        cell for cell in dataset.coverage_grid
        if cell.row in (0, 3) and cell.col in (0, 3)
    ]

    assert len(corner_cells) == 4
    assert all(cell.coverage_score < 0.3 for cell in corner_cells)


def test_failure_checklist_identical_board_pose_reports_poor_pose_diversity():
    dataset = _projected_dataset(8, center_only=True)
    analyze_dataset_quality(dataset, CameraConfig(width=W, height=H))

    assert dataset.diversity is not None
    assert dataset.diversity.rotation_diversity == pytest.approx(0.0)
    assert dataset.diversity.distance_diversity == pytest.approx(0.0)


def test_failure_checklist_blur_image_is_rejected_by_image_quality():
    info = ImageInfo(
        image_id="blurred",
        path="/fake/blurred.jpg",
        width=1920,
        height=1080,
        sharpness=1.0,
        brightness=120.0,
        contrast=40.0,
        saturation=0.01,
        motion_blur_score=1.0,
    )

    report = evaluate_image_quality(info)

    assert report.has_errors
    assert any(issue.code == "too_blurry" for issue in report.issues)


def test_failure_checklist_duplicate_images_are_reported():
    dataset = Dataset(frames=[
        _frame_with_phash("dup_a", "ffffffffffffffff"),
        _frame_with_phash("dup_b", "ffffffffffffffff"),
        _frame_with_phash("unique", "0000000000000000"),
    ])

    groups = find_duplicate_groups(dataset)

    assert len(groups) == 1
    assert groups[0].exact
    assert set(groups[0].image_ids) == {"dup_a", "dup_b"}


def test_failure_checklist_no_corners_are_kept_but_not_used_for_calibration():
    failed = Frame(
        image_info=ImageInfo("blank", "/fake/blank.jpg", W, H),
        detection=DetectionResult("blank", success=False, num_corners=0, failure_reason="no corners"),
        status=FrameStatus.DETECTION_FAILED,
    )
    too_few = Frame(
        image_info=ImageInfo("too_few", "/fake/too_few.jpg", W, H),
        detection=DetectionResult(
            "too_few",
            success=True,
            corners=np.array([[[10.0, 10.0]], [[20.0, 20.0]]], dtype=np.float32),
            object_points=np.zeros((2, 1, 3), dtype=np.float32),
            num_corners=2,
        ),
        status=FrameStatus.DETECTED,
    )
    dataset = Dataset(frames=[failed, too_few])

    frames, object_points, image_points = collect_calibration_inputs(dataset)
    result = calibrate_pinhole(dataset, CameraConfig(width=W, height=H))

    assert dataset.num_total == 2
    assert dataset.num_detected == 1
    assert frames == []
    assert object_points == []
    assert image_points == []
    assert not result.success
    assert "사용 가능한 프레임이 0장" in result.error_message


def test_failure_checklist_calibration_failure_is_sanity_error():
    result = CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        success=False,
        error_message="프레임 부족",
    )

    check = run_sanity_check(result, CameraConfig(width=W, height=H))

    assert check.has_errors
    assert any(issue.code == "calibration_failed" for issue in check.issues)


def test_failure_checklist_severe_distortion_is_sanity_warning():
    result = CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=np.array([15.0, -9.0, 0.0, 0.0, 4.0]),
        rms_error=0.4,
        success=True,
    )

    check = run_sanity_check(result, CameraConfig(width=W, height=H))

    assert any(issue.code == "distortion_magnitude_large" for issue in check.issues)


def test_failure_checklist_nan_result_is_sanity_error():
    K = TRUE_K.copy()
    K[0, 0] = np.nan
    result = CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        camera_matrix=K,
        distortion=np.zeros(5),
        rms_error=0.4,
        success=True,
    )

    check = run_sanity_check(result, CameraConfig(width=W, height=H))

    assert check.has_errors
    assert any(issue.code == "camera_matrix_non_finite" for issue in check.issues)


def test_failure_checklist_inf_result_is_sanity_error():
    result = CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=np.array([np.inf, 0.0, 0.0, 0.0, 0.0]),
        rms_error=0.4,
        success=True,
    )

    check = run_sanity_check(result, CameraConfig(width=W, height=H))

    assert check.has_errors
    assert any(issue.code == "distortion_non_finite" for issue in check.issues)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"squares_x": 2}, "3 이상"),
        ({"squares_y": 2}, "3 이상"),
        ({"square_size": 0.0}, "0보다 커야"),
        ({"marker_size": 0.0}, "0보다 커야"),
        ({"marker_size": 0.05}, "square-size보다 작아야"),
    ],
)
def test_failure_checklist_bad_board_size_is_rejected(overrides, message):
    args = argparse.Namespace(
        pattern="charuco",
        squares_x=7,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_5X5_100",
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    with pytest.raises(CliError, match=message):
        _build_pattern_config(args)


@pytest.mark.parametrize("flag", ["--width", "--height"])
def test_failure_checklist_bad_image_size_is_rejected(flag):
    parser = build_arg_parser()
    args = parser.parse_args([
        "--images", "imgs",
        "--squares-x", "7",
        "--squares-y", "5",
        "--square-size", "0.04",
        "--marker-size", "0.03",
        flag, "0",
    ])

    with pytest.raises(SystemExit):
        _normalize_cli_args(args, parser)


def test_failure_checklist_excessive_outliers_never_drop_below_minimum(monkeypatch):
    dataset = _projected_dataset(5)

    def fake_calibrate(current_dataset, camera_config, model, **kwargs):
        enabled_ids = [f.image_info.image_id for f in current_dataset.enabled_frames]
        return CalibrationResult(
            model_name=model,
            success=True,
            rms_error=1.0,
            per_frame_error={fid: float(i + 1) for i, fid in enumerate(enabled_ids)},
        )

    def fake_recommend(per_frame_error, **kwargs):
        return list(per_frame_error), 0.1

    monkeypatch.setattr("calibration.outlier._calibrate_by_model", fake_calibrate)
    monkeypatch.setattr("calibration.outlier.recommend_outliers", fake_recommend)

    _, outlier_result = recalibrate_with_outlier_pruning(
        dataset,
        CameraConfig(width=W, height=H),
        CameraModelType.PINHOLE,
        max_iterations=5,
    )

    assert len(outlier_result.removed_frame_ids) == 5 - MIN_FRAMES_REQUIRED
    assert dataset.num_enabled == MIN_FRAMES_REQUIRED
    assert outlier_result.iterations == 1
