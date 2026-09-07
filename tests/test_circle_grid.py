from __future__ import annotations

import cv2
import numpy as np

from calibration.detector import (
    build_circle_grid_blob_detector,
    build_circle_grid_object_points,
    build_detect_fn,
    detect_circle_grid,
    maximum_pattern_corners,
)
from calibration.types import CircleGridType, PatternConfig, PatternType


def _pattern(
    grid_type: CircleGridType = CircleGridType.SYMMETRIC,
    squares_x: int = 4,
    squares_y: int = 3,
) -> PatternConfig:
    return PatternConfig(
        type=PatternType.CIRCLE_GRID,
        squares_x=squares_x,
        squares_y=squares_y,
        square_size=0.02,
        circle_grid_type=grid_type,
    )


def _render_circle_grid(
    pattern: PatternConfig, px_spacing: int = 90, invert: bool = False
) -> np.ndarray:
    """합성 circle grid 이미지를 만든다.

    invert=False(기본): 검은 원 / 밝은 배경 (실사용에서 가장 흔한 인쇄물
    패턴). invert=True: 밝은 원 / 어두운 배경 - 반전된 패턴(백라이트
    모니터 표시, 네거티브 인쇄 등)도 검출돼야 한다는 걸 확인하기 위함.
    """
    points = build_circle_grid_object_points(pattern).reshape(-1, 3)
    scale = px_spacing / pattern.square_size
    xy = points[:, :2] * scale
    margin = 80
    w = int(np.max(xy[:, 0]) + margin * 2 + 1)
    h = int(np.max(xy[:, 1]) + margin * 2 + 1)
    background = 0 if invert else 255
    foreground = 255 if invert else 0
    image = np.full((h, w), background, dtype=np.uint8)
    for x, y in xy:
        cv2.circle(image, (int(round(x + margin)), int(round(y + margin))), 22, foreground, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_symmetric_circle_grid_object_points_are_row_major():
    obj = build_circle_grid_object_points(_pattern()).reshape(-1, 3)
    assert obj.shape == (12, 3)
    assert np.allclose(obj[0], [0.0, 0.0, 0.0])
    assert np.allclose(obj[1], [0.02, 0.0, 0.0])
    assert np.allclose(obj[4], [0.0, 0.02, 0.0])


def test_asymmetric_circle_grid_object_points_use_opencv_row_offset():
    obj = build_circle_grid_object_points(_pattern(CircleGridType.ASYMMETRIC)).reshape(-1, 3)
    assert np.allclose(obj[0], [0.0, 0.0, 0.0])
    assert np.allclose(obj[1], [0.04, 0.0, 0.0])
    assert np.allclose(obj[4], [0.02, 0.02, 0.0])
    assert np.allclose(obj[5], [0.06, 0.02, 0.0])


def test_circle_grid_dispatch_and_maximum_count():
    pattern = _pattern()
    assert maximum_pattern_corners(pattern) == 12
    detect_fn = build_detect_fn(pattern)
    result = detect_fn(_render_circle_grid(pattern), "circle_grid")
    assert result.success, result.failure_reason
    assert result.corners.shape == (12, 1, 2)
    assert result.object_points.shape == (12, 1, 3)


def test_asymmetric_circle_grid_detection_shape():
    # 4x3(12점)은 OpenCV의 CALIB_CB_ASYMMETRIC_GRID 알고리즘 자체가 blob
    # polarity/CLUSTERING 설정과 무관하게 검출하지 못하는 크기임을 로컬에서
    # 재현/확인했다(실사용 asymmetric 보드도 이 정도로 작게 쓰지 않는다 -
    # OpenCV 공식 샘플 패턴도 4x11이다). 그래서 이 shape 검증 테스트는
    # 실제로 검출 가능한 4x5(20점)를 쓴다. object point 순서/오프셋
    # 자체는 test_asymmetric_circle_grid_object_points_use_opencv_row_offset
    # 가 4x3으로 이미 별도 검증한다.
    pattern = _pattern(CircleGridType.ASYMMETRIC, squares_x=4, squares_y=5)
    result = detect_circle_grid(_render_circle_grid(pattern), pattern, "asymmetric")
    assert result.success, result.failure_reason
    assert result.num_corners == 20


def test_symmetric_circle_grid_detection_dark_circles_on_light_background():
    pattern = _pattern(CircleGridType.SYMMETRIC)
    image = _render_circle_grid(pattern, invert=False)
    result = detect_circle_grid(image, pattern, "symmetric_dark")
    assert result.success, result.failure_reason
    assert result.num_corners == 12


def test_symmetric_circle_grid_detection_light_circles_on_dark_background():
    pattern = _pattern(CircleGridType.SYMMETRIC)
    image = _render_circle_grid(pattern, invert=True)
    result = detect_circle_grid(image, pattern, "symmetric_light")
    assert result.success, result.failure_reason
    assert result.num_corners == 12


def test_asymmetric_circle_grid_detection_dark_circles_on_light_background():
    pattern = _pattern(CircleGridType.ASYMMETRIC, squares_x=4, squares_y=5)
    image = _render_circle_grid(pattern, invert=False)
    result = detect_circle_grid(image, pattern, "asymmetric_dark")
    assert result.success, result.failure_reason
    assert result.num_corners == 20


def test_asymmetric_circle_grid_detection_light_circles_on_dark_background():
    pattern = _pattern(CircleGridType.ASYMMETRIC, squares_x=4, squares_y=5)
    image = _render_circle_grid(pattern, invert=True)
    result = detect_circle_grid(image, pattern, "asymmetric_light")
    assert result.success, result.failure_reason
    assert result.num_corners == 20


def test_circle_grid_custom_blob_detector_is_used_as_is():
    """호출자가 blob_detector를 직접 넘기면 내부 polarity/CLUSTERING
    fallback을 타지 않고 그 detector를 그대로 존중해야 한다(기존 API
    계약). 여기서는 실제 이미지 극성과 일치하는 detector를 넘겨서 단일
    시도로도 성공하는 것만 확인한다 - fallback 없이도 동작해야 한다는 뜻."""
    pattern = _pattern(CircleGridType.SYMMETRIC)
    image = _render_circle_grid(pattern, invert=False)
    detector = build_circle_grid_blob_detector(blob_color=0)
    result = detect_circle_grid(image, pattern, "custom_detector", blob_detector=detector)
    assert result.success, result.failure_reason
