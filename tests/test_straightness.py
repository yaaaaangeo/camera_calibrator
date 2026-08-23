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
import pytest

from calibration.straightness import (
    compute_straightness_breakdown,
    compute_straightness_residual,
    format_straightness_breakdown,
    format_straightness_summary,
    _classify_position,
)
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


# ---------------------------------------------------------------------------
# 설계 문서 15번 - Line Straightness 평가 강화 (방향/위치별 분해)
# ---------------------------------------------------------------------------

class TestClassifyPosition:
    def test_middle_index_is_center(self):
        assert _classify_position(3, 7) == "center"

    def test_first_index_is_edge(self):
        assert _classify_position(0, 7) == "edge"

    def test_last_index_is_edge(self):
        assert _classify_position(6, 7) == "edge"

    def test_tiny_total_is_always_edge(self):
        assert _classify_position(0, 2) == "edge"
        assert _classify_position(1, 1) == "edge"


class TestComputeStraightnessBreakdown:
    def test_overall_matches_scalar_residual(self, pattern_config):
        """breakdown.overall_error는 항상 compute_straightness_residual()과
        정확히 같아야 한다 - 두 함수가 같은 라인 수집 로직을 공유하기 때문."""
        frame = _synthetic_charuco_frame(pattern_config)
        scalar_residual, n_lines = compute_straightness_residual(
            [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
        )
        breakdown = compute_straightness_breakdown(
            [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
        )
        assert breakdown.overall_error == pytest.approx(scalar_residual)
        assert breakdown.num_lines == n_lines

    def test_breakdown_has_horizontal_and_vertical(self, pattern_config):
        frame = _synthetic_charuco_frame(pattern_config)
        breakdown = compute_straightness_breakdown(
            [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
        )
        assert breakdown.horizontal_error is not None
        assert breakdown.vertical_error is not None
        assert breakdown.num_lines > 0

    def test_empty_frames_returns_zero_lines(self, pattern_config):
        breakdown = compute_straightness_breakdown([], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE)
        assert breakdown.num_lines == 0
        assert breakdown.overall_error is None

    def test_wrong_distortion_gives_worse_edge_than_correct_distortion(self, pattern_config):
        """왜곡 계수를 아예 0으로 두고(보정 안 한 것과 같음) 계산하면, 실제
        계수로 계산했을 때보다 edge_line_error가 더 나빠야(커야) 한다 -
        방사 왜곡은 외곽에서 더 심하게 나타나기 때문."""
        frame = _synthetic_charuco_frame(pattern_config)
        correct = compute_straightness_breakdown(
            [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
        )
        wrong = compute_straightness_breakdown(
            [frame], pattern_config, TRUE_K, ZERO_D, CameraModelType.PINHOLE
        )
        assert wrong.edge_line_error > correct.edge_line_error


class TestFormatStraightnessBreakdown:
    def test_includes_all_categories(self, pattern_config):
        frame = _synthetic_charuco_frame(pattern_config)
        breakdown = compute_straightness_breakdown(
            [frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
        )
        text = format_straightness_breakdown(breakdown)
        for label in ("Horizontal", "Vertical", "Diagonal", "Center line", "Edge line", "Corner line", "Overall"):
            assert label in text

    def test_handles_empty(self):
        from calibration.types import StraightnessBreakdown
        text = format_straightness_breakdown(StraightnessBreakdown(num_lines=0))
        assert "부족합니다" in text
