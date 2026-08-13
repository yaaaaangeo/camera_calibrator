"""
tests/test_straightness.py
===============================

설계 문서 3.4번 - Line Straightness Residual. ChArUco 격자 자체가 실세계에서
직선이라는 사실을 이용해, 왜곡 보정이 정확할수록 undistort된 좌표가 실제로
더 직선에 가까워지는지 검증한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.straightness import compute_straightness_residual, format_straightness_summary
from calibration.types import CameraModelType, DetectionResult, Frame, FrameStatus, ImageInfo

W, H = 1920, 1080
TRUE_K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])
ZERO_D = np.zeros(5)


def _synthetic_charuco_frame(pattern_config):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern_config.squares_x, pattern_config.squares_y),
        pattern_config.square_size, pattern_config.marker_size, aruco_dict,
    )
    pts3d = board.getChessboardCorners().astype(np.float32)

    rvec = np.array([0.15, -0.1, 0.05])
    tvec = np.array([0.05, -0.03, 0.5])
    distorted_img_pts, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, TRUE_K, TRUE_D)
    distorted_img_pts = distorted_img_pts.reshape(-1, 2).astype(np.float32)

    n = pts3d.shape[0]
    ids = np.arange(n, dtype=np.int32).reshape(-1, 1)
    info = ImageInfo(image_id="synth", path="-", width=W, height=H)
    det = DetectionResult(
        image_id="synth", success=True,
        corners=distorted_img_pts.reshape(-1, 1, 2),
        object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=n,
    )
    return Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)


def test_correct_distortion_gives_near_zero_residual(pattern_config):
    """진짜 왜곡 계수로 undistort하면 격자가 거의 완벽한 직선이 되어야 한다."""
    frame = _synthetic_charuco_frame(pattern_config)
    residual, n_lines = compute_straightness_residual(
        [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
    )
    assert residual is not None
    assert n_lines > 0
    assert residual < 0.05, f"올바른 왜곡계수로 보정했는데 잔차가 너무 큼: {residual}"


def test_ignoring_distortion_gives_larger_residual(pattern_config):
    """왜곡을 무시(0)하고 undistort하면 여전히 휘어 있어야 한다 - 잔차가
    올바른 보정 대비 뚜렷하게 커야 한다.
    """
    frame = _synthetic_charuco_frame(pattern_config)
    residual_correct, _ = compute_straightness_residual(
        [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
    )
    residual_wrong, _ = compute_straightness_residual(
        [frame], pattern_config, TRUE_K, ZERO_D, CameraModelType.PINHOLE
    )
    assert residual_wrong is not None
    assert residual_wrong > residual_correct * 5, (
        "왜곡을 무시하면 직선성 잔차가 정확한 보정 대비 뚜렷하게 커야 함"
    )


def test_fisheye_path_no_crash(pattern_config):
    frame = _synthetic_charuco_frame(pattern_config)
    D4 = np.zeros(4)
    residual, n_lines = compute_straightness_residual(
        [frame], pattern_config, TRUE_K, D4, CameraModelType.FISHEYE
    )
    assert n_lines > 0


def test_insufficient_points_returns_none(pattern_config):
    """코너가 min_points_per_line보다 적으면 계산할 라인이 없어 None을 반환해야 한다."""
    info = ImageInfo(image_id="tiny", path="-", width=W, height=H)
    det = DetectionResult(
        image_id="tiny", success=True,
        corners=np.array([[100, 100], [200, 100]], dtype=np.float32).reshape(-1, 1, 2),
        object_points=np.zeros((2, 1, 3), dtype=np.float32),
        ids=np.array([[0], [1]], dtype=np.int32),
        num_corners=2,
    )
    frame = Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)
    residual, n_lines = compute_straightness_residual(
        [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
    )
    assert residual is None
    assert n_lines == 0


def test_format_straightness_summary_no_crash():
    assert "px" in format_straightness_summary(0.42, 10)
    assert format_straightness_summary(None, 0) is not None
