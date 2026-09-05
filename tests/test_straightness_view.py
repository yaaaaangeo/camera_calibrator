"""
tests/test_straightness_view.py
====================================

설계 문서 3.4번 확장 - Straightness Residual을 숫자 하나가 아니라 이미지
위에 겹쳐서 보여주는 기능. 두 층을 나눠서 검증한다:

1. calibration/straightness.py의 새 함수(compute_frame_straightness_lines,
   sort_points_along_line) - 백엔드 계산이 정확한지.
2. ui/straightness_view.py - 실제로 화면에 그려지고, "왜곡을 무시한 모델"과
   "제대로 모델링한 모델"의 시각적 차이가 뚜렷한지(초록 vs 빨강 비율).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.straightness import (
    compute_frame_straightness_lines,
    compute_straightness_residual,
    sort_points_along_line,
)
from calibration.types import CameraModelType, DetectionResult, Frame, FrameStatus, ImageInfo

pytestmark = pytest.mark.slow

W, H = 1920, 1080
TRUE_K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])
ZERO_D = np.zeros(5)


def _synthetic_frame(pattern_config):
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


def test_sort_points_along_line_orders_by_position():
    pts = np.array([[3, 3], [0, 0], [2, 2], [1, 1], [4, 4]], dtype=float)
    sorted_pts = sort_points_along_line(pts)
    xs = sorted_pts[:, 0]
    assert list(xs) == sorted(xs) or list(xs) == sorted(xs, reverse=True)


def test_frame_lines_match_aggregate_exactly(pattern_config):
    """compute_frame_straightness_lines()로 만든 라인들의 평균이 기존
    compute_straightness_residual() 결과와 완전히 같아야 한다.
    """
    frame = _synthetic_frame(pattern_config)

    lines = compute_frame_straightness_lines(frame, pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE)
    assert len(lines) > 0

    manual_avg = float(np.mean([l.residual for l in lines]))
    agg, n = compute_straightness_residual([frame], pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE)

    assert n == len(lines)
    assert abs(agg - manual_avg) < 1e-9


def test_lines_reveal_which_specific_line_is_worse(pattern_config):
    """이게 이 기능의 핵심 가치 - "전체 평균 하나"가 아니라 "어느 줄이
    문제인지" 알 수 있어야 한다.
    """
    frame = _synthetic_frame(pattern_config)
    lines = compute_frame_straightness_lines(frame, pattern_config, TRUE_K, ZERO_D, CameraModelType.PINHOLE)

    residuals = [l.residual for l in lines]
    assert max(residuals) > min(residuals) * 2, "모든 줄의 잔차가 비슷하면 시각화 정보 가치가 없음"


def test_target_k_changes_pixel_coordinates_but_not_residual_direction(pattern_config):
    """target_K를 다른 값으로 주면 undistort된 좌표값 자체는 달라지지만
    (fisheye의 재추정된 K와 화면을 맞추기 위한 용도), "왜곡 무시가 더
    나쁘다"는 상대적 결론 자체는 안 바뀌어야 한다.
    """
    frame = _synthetic_frame(pattern_config)
    alt_K = TRUE_K.copy()
    alt_K[0, 0] *= 1.5

    lines_default = compute_frame_straightness_lines(
        frame, pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE
    )
    lines_alt_k = compute_frame_straightness_lines(
        frame, pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE, target_K=alt_K
    )

    assert not np.allclose(lines_default[0].points, lines_alt_k[0].points)
    assert np.mean([l.residual for l in lines_default]) < 0.05
    assert np.mean([l.residual for l in lines_alt_k]) < 0.05


def test_insufficient_points_returns_empty_list(pattern_config):
    info = ImageInfo(image_id="tiny", path="-", width=W, height=H)
    det = DetectionResult(
        image_id="tiny", success=True,
        corners=np.array([[100, 100], [200, 100]], dtype=np.float32).reshape(-1, 1, 2),
        object_points=np.zeros((2, 1, 3), dtype=np.float32),
        ids=np.array([[0], [1]], dtype=np.int32),
        num_corners=2,
    )
    frame = Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)
    lines = compute_frame_straightness_lines(frame, pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE)
    assert lines == []


# ---------------------------------------------------------------------------
# UI 레이어
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_render_overlay_produces_valid_image(qapp, pattern_config):
    from ui.straightness_view import render_straightness_overlay

    frame = _synthetic_frame(pattern_config)
    blank_canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    canvas, lines = render_straightness_overlay(
        blank_canvas, frame, pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE, TRUE_K,
    )
    assert canvas.shape == blank_canvas.shape
    assert len(lines) > 0
    assert not np.all(canvas == 255)


def test_ignoring_distortion_produces_more_red_pixels_than_correct_model(qapp, pattern_config):
    """이 기능의 최종 목적 - 왜곡을 무시한(더 휜) 결과가 실제로 화면에
    더 "빨간" 픽셀로 나타나야 한다.
    """
    from ui.straightness_view import render_straightness_overlay

    frame = _synthetic_frame(pattern_config)
    blank_canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    canvas_correct, _ = render_straightness_overlay(
        blank_canvas.copy(), frame, pattern_config, TRUE_K, TRUE_D, CameraModelType.PINHOLE, TRUE_K,
    )
    canvas_wrong, _ = render_straightness_overlay(
        blank_canvas.copy(), frame, pattern_config, TRUE_K, ZERO_D, CameraModelType.PINHOLE, TRUE_K,
    )

    def redness(canvas):
        mask = np.any(canvas != 255, axis=2)
        if not mask.any():
            return 0.0
        pixels = canvas[mask].astype(int)
        return float(np.mean(pixels[:, 2] - pixels[:, 1]))  # R - G (BGR이므로)

    redness_correct = redness(canvas_correct)
    redness_wrong = redness(canvas_wrong)
    assert redness_wrong > redness_correct, (
        f"왜곡을 무시한 결과가 더 붉어야 함 (correct={redness_correct:.1f}, wrong={redness_wrong:.1f})"
    )
