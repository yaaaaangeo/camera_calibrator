"""
tests/test_windshield_base_projection.py
=============================================

calibration.windshield.base_projection::solve_poses_fixed_intrinsics 검증 -
특히 "Base K,D는 절대 재최적화하지 않는다"는 핵심 원칙(사용자 스펙 3번)을
코드 레벨에서 확인한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import CameraModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics
from tests._windshield_test_utils import build_synthetic_windshield_dataset, default_camera_matrix_distortion


def test_solve_poses_does_not_mutate_base_intrinsics():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_windshield_dataset(K, D)

    solve_poses_fixed_intrinsics(dataset.frames, K, D, CameraModelType.BROWN_CONRADY)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_solve_poses_returns_parallel_aligned_lists():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)

    ok_frames, rvecs, tvecs, failed = solve_poses_fixed_intrinsics(
        dataset.frames, K, D, CameraModelType.BROWN_CONRADY
    )

    assert len(ok_frames) == len(dataset.frames)
    assert len(rvecs) == len(ok_frames)
    assert len(tvecs) == len(ok_frames)
    assert failed == []


def test_solve_poses_reports_failed_frame_without_crashing_batch():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    # 하나는 검출 실패로 만든다 - solve_poses_fixed_intrinsics는 예외를 던지지
    # 않고 failed_frame_ids에만 담아야 한다.
    bad_frame = dataset.frames[0]
    bad_frame.detection.success = False

    ok_frames, rvecs, tvecs, failed = solve_poses_fixed_intrinsics(
        dataset.frames, K, D, CameraModelType.BROWN_CONRADY
    )

    assert bad_frame.image_info.image_id in failed
    assert len(ok_frames) == len(dataset.frames) - 1
    assert len(rvecs) == len(ok_frames) == len(tvecs)


def test_solve_poses_accepts_fisheye_charuco_point_shapes():
    """Windshield detection stores ChArUco points as Nx3/Nx1x2, while
    cv2.fisheye.solvePnP requires normalized Nx1x3/Nx1x2 arrays."""
    K, _ = default_camera_matrix_distortion()
    D = np.array([[-0.08], [0.01], [0.0], [0.0]], dtype=np.float64)
    dataset = build_synthetic_windshield_dataset(K, np.zeros(5))
    frame = dataset.frames[0]
    obj = np.asarray(frame.detection.object_points, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.array([[0.08], [-0.04], [0.02]], dtype=np.float64)
    tvec = np.array([[0.05], [-0.03], [2.5]], dtype=np.float64)
    corners, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    frame.detection.corners = corners.astype(np.float32)

    ok_frames, rvecs, tvecs, failed = solve_poses_fixed_intrinsics(
        [frame], K, D, CameraModelType.FISHEYE
    )

    assert len(ok_frames) == len(rvecs) == len(tvecs) == 1
    assert failed == []
