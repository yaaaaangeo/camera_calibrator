"""
tests/test_detector.py
===========================

설계 문서 17번 Step2 - ChArUco Detection. 검출 성공/실패 상태가 프레임에
정확히 기록되는지, 실패해도(설계 문서: "검출 실패 이미지도 바로 삭제하지
않고 status만 기록") 파이프라인 전체가 죽지 않는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import cv2

from calibration.detector import build_charuco_board, detect_dataset, summarize_dataset
from calibration.types import FrameStatus, PatternConfig, PatternType


def test_builds_7x7_charuco_board_with_7x7_dictionary():
    pattern = PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=7,
        squares_y=7,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_7X7_100",
    )

    board = build_charuco_board(pattern)

    assert board.getChessboardSize() == (7, 7)
    assert board.getDictionary().markerSize == 7
    assert len(board.getChessboardCorners()) == 36


def test_detect_dataset_succeeds_on_real_charuco_images(synthetic_distorted_dataset_dir, pattern_config):
    import glob
    paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))
    dataset = detect_dataset(paths, pattern_config)

    assert dataset.num_total == len(paths)
    assert dataset.num_detected >= 10
    for f in dataset.frames:
        if f.detection and f.detection.success:
            assert f.status == FrameStatus.DETECTED
            assert f.detection.num_corners > 0
            assert f.detection.corners is not None
            assert f.detection.ids is not None


def test_detect_dataset_handles_blank_image_without_crashing(tmp_path, pattern_config):
    """보드가 전혀 없는 흰 이미지는 검출 실패로 기록돼야지, 파이프라인
    전체가 죽으면 안 된다 (설계 문서: 실패 이미지도 삭제 없이 status만 기록).
    """
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    path = str(tmp_path / "blank.jpg")
    cv2.imwrite(path, blank)

    dataset = detect_dataset([path], pattern_config)
    assert dataset.num_total == 1
    assert dataset.num_detected == 0
    frame = dataset.frames[0]
    assert frame.status == FrameStatus.DETECTION_FAILED
    assert frame.image_info.path == path  # 파일 자체는 그대로 참조되어야 함


def test_detect_dataset_reports_progress_for_every_image(tmp_path, pattern_config):
    paths = []
    for index in range(3):
        path = str(tmp_path / f"blank_{index}.jpg")
        cv2.imwrite(path, np.full((32, 48, 3), 255, dtype=np.uint8))
        paths.append(path)

    calls = []
    detect_dataset(
        paths,
        pattern_config,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_detect_dataset_mixed_success_and_failure(synthetic_distorted_dataset_dir, tmp_path, pattern_config):
    """일부는 성공, 일부는 실패인 혼합 데이터셋에서도 각 프레임 상태가
    독립적으로 정확히 기록돼야 한다.
    """
    import glob
    good_paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))[:3]

    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    blank_path = str(tmp_path / "blank_mixed.jpg")
    cv2.imwrite(blank_path, blank)

    dataset = detect_dataset(good_paths + [blank_path], pattern_config)
    assert dataset.num_total == 4
    statuses = {f.image_info.path: f.status for f in dataset.frames}
    assert statuses[blank_path] == FrameStatus.DETECTION_FAILED
    assert sum(1 for s in statuses.values() if s == FrameStatus.DETECTED) == 3


def test_summarize_dataset_no_crash(synthetic_dataset):
    text = summarize_dataset(synthetic_dataset)
    assert len(text) > 0
    assert "검출" in text or "%" in text


def test_partial_charuco_detection_recovers_true_board_footprint(charuco_board):
    """Test 2 (계획 문서 11번) - 같은 실제 board pose에서 일부 corner만
    보이도록 만든 뒤, full detection과 partial detection에서 추정되는
    board_center_px/board_area_ratio가 크게 뒤집히지 않는지 확인한다.

    검출된 코너만으로 convexHull/mean을 구하는(옛 방식) 대신, solvePnP로
    구한 pose에 전체 보드 코너를 투영해 footprint를 복원하면(Problem 6)
    partial detection에서도 full detection과 훨씬 가까운 값이 나와야 한다.
    """
    from calibration.detector import _compute_board_geometry, _estimate_full_board_footprint
    from calibration.models.common import rough_camera_matrix

    img_w, img_h = 1920, 1080
    full_object_points = charuco_board.getChessboardCorners().astype(np.float64)
    n = full_object_points.shape[0]
    squares_x, _squares_y = charuco_board.getChessboardSize()
    n_cols = squares_x - 1

    # 실제 보드가 화면 오른쪽으로 치우치고 기울어진 pose - 오른쪽 절반만
    # 보이는 partial detection을 시뮬레이션하기 좋은 조건.
    rvec = np.radians(np.array([0.0, 25.0, 0.0])).reshape(3, 1)
    tvec = np.array([[0.15], [0.0], [1.0]])
    K = rough_camera_matrix((img_w, img_h))
    full_image_points, _ = cv2.projectPoints(full_object_points, rvec, tvec, K, None)
    full_image_points = full_image_points.reshape(-1, 2)

    full_area, full_center, _, _ = _compute_board_geometry(
        full_image_points.reshape(-1, 1, 2).astype(np.float32), (img_h, img_w)
    )

    # Partial detection: 격자의 왼쪽 절반 칼럼만 "검출된" 코너로 남긴다
    # (오른쪽 절반이 화면 밖/가려짐으로 검출 실패했다고 가정).
    keep_mask = np.array([(i % n_cols) < n_cols // 2 for i in range(n)])
    assert 0 < keep_mask.sum() < n

    partial_object_points = full_object_points[keep_mask].reshape(-1, 1, 3).astype(np.float64)
    partial_image_points = full_image_points[keep_mask].reshape(-1, 1, 2).astype(np.float32)

    naive_area, naive_center, _, _ = _compute_board_geometry(
        partial_image_points, (img_h, img_w)
    )

    footprint = _estimate_full_board_footprint(
        partial_object_points, partial_image_points, full_object_points, (img_h, img_w)
    )
    assert footprint is not None
    recovered_area, recovered_center = footprint

    naive_center_error = float(np.hypot(naive_center[0] - full_center[0], naive_center[1] - full_center[1]))
    recovered_center_error = float(
        np.hypot(recovered_center[0] - full_center[0], recovered_center[1] - full_center[1])
    )
    assert recovered_center_error < naive_center_error
    # The naive convexHull-of-detected-only estimate must actually be biased
    # (otherwise this test wouldn't be exercising the failure mode at all).
    assert naive_center_error > 20.0
    # And the reconstruction must land close to the true full-board center.
    assert recovered_center_error < 15.0

    assert abs(recovered_area - full_area) < abs(naive_area - full_area)


def test_full_board_footprint_falls_back_to_none_on_insufficient_points():
    from calibration.detector import _estimate_full_board_footprint

    footprint = _estimate_full_board_footprint(
        np.zeros((2, 1, 3)), np.zeros((2, 1, 2)), np.zeros((10, 3)), (1080, 1920)
    )
    assert footprint is None
