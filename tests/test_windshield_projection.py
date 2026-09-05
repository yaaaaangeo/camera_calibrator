"""
tests/test_windshield_projection.py
========================================

BaselineWindshieldModel의 런타임 Projection API(project_point/unproject_pixel,
사용자 스펙 17번) 검증. Baseline은 보정이 항등이므로, project_point는
cv2.projectPoints/cv2.fisheye.projectPoints와 정확히 같아야 하고,
unproject_pixel -> project_point는 원래 픽셀로 되돌아와야 한다(round-trip).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.windshield.baseline import BaselineWindshieldModel


def test_project_point_matches_cv2_projectpoints_for_pinhole_family():
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    model = BaselineWindshieldModel(K, D, CameraModelType.BROWN_CONRADY)

    point = (0.05, -0.03, 0.8)
    u, v = model.project_point(*point)

    expected, _ = cv2.projectPoints(
        np.array([[point]], dtype=np.float64), np.zeros(3), np.zeros(3), K, D
    )
    ex, ey = expected.reshape(-1)
    assert u == pytest.approx(ex, abs=1e-6)
    assert v == pytest.approx(ey, abs=1e-6)


def test_project_point_matches_cv2_projectpoints_for_fisheye():
    K = np.array([[500.0, 0.0, 640.0], [0.0, 500.0, 400.0], [0.0, 0.0, 1.0]])
    D = np.array([[0.01], [0.001], [0.0001], [0.0]])
    model = BaselineWindshieldModel(K, D, CameraModelType.FISHEYE)

    point = (0.1, 0.05, 1.0)
    u, v = model.project_point(*point)

    expected, _ = cv2.fisheye.projectPoints(
        np.array([[point]], dtype=np.float64).reshape(1, -1, 3), np.zeros(3), np.zeros(3), K, D
    )
    ex, ey = expected.reshape(-1)
    assert u == pytest.approx(ex, abs=1e-6)
    assert v == pytest.approx(ey, abs=1e-6)


@pytest.mark.parametrize(
    "model_type,K,D,point",
    [
        (
            CameraModelType.BROWN_CONRADY,
            np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]]),
            np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]]),
            (0.05, -0.03, 0.8),
        ),
        (
            CameraModelType.FISHEYE,
            np.array([[500.0, 0.0, 640.0], [0.0, 500.0, 400.0], [0.0, 0.0, 1.0]]),
            np.array([[0.01], [0.001], [0.0001], [0.0]]),
            (0.1, 0.05, 1.0),
        ),
    ],
)
def test_unproject_then_project_round_trips_to_same_pixel(model_type, K, D, point):
    model = BaselineWindshieldModel(K, D, model_type)
    u, v = model.project_point(*point)

    dx, dy, dz = model.unproject_pixel(u, v)
    # 단위 광선 방향 * depth로 원래 3D 점을 복원(z 성분을 맞추는 스케일).
    depth = point[2] / dz
    recon = (dx * depth, dy * depth, dz * depth)

    u2, v2 = model.project_point(*recon)
    assert u2 == pytest.approx(u, abs=1e-4)
    assert v2 == pytest.approx(v, abs=1e-4)


def test_unproject_pixel_returns_unit_vector():
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    model = BaselineWindshieldModel(K, D, CameraModelType.BROWN_CONRADY)

    dx, dy, dz = model.unproject_pixel(700.0, 380.0)
    norm = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)
