from __future__ import annotations

import cv2
import numpy as np

from calibration.detector import (
    build_circle_grid_object_points,
    build_detect_fn,
    detect_circle_grid,
    maximum_pattern_corners,
)
from calibration.types import CircleGridType, PatternConfig, PatternType


def _pattern(grid_type: CircleGridType = CircleGridType.SYMMETRIC) -> PatternConfig:
    return PatternConfig(
        type=PatternType.CIRCLE_GRID,
        squares_x=4,
        squares_y=3,
        square_size=0.02,
        circle_grid_type=grid_type,
    )


def _render_circle_grid(pattern: PatternConfig, px_spacing: int = 90) -> np.ndarray:
    points = build_circle_grid_object_points(pattern).reshape(-1, 3)
    scale = px_spacing / pattern.square_size
    xy = points[:, :2] * scale
    margin = 80
    w = int(np.max(xy[:, 0]) + margin * 2 + 1)
    h = int(np.max(xy[:, 1]) + margin * 2 + 1)
    image = np.full((h, w), 255, dtype=np.uint8)
    for x, y in xy:
        cv2.circle(image, (int(round(x + margin)), int(round(y + margin))), 22, 0, -1)
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
    pattern = _pattern(CircleGridType.ASYMMETRIC)
    result = detect_circle_grid(_render_circle_grid(pattern), pattern, "asymmetric")
    assert result.success, result.failure_reason
    assert result.num_corners == 12
