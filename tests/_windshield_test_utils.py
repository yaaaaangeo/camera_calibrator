"""
tests._windshield_test_utils
=================================

Windshield 테스트 전용 헬퍼 - 실제 ChArUco 이미지 렌더링/검출 없이, 알려진
K,D,pose로 noiseless 코너를 직접 생성한 최소 Dataset을 만든다. Baseline은
"관측 픽셀 - 예측 픽셀"만 비교하므로 실제 이미지/검출기가 전혀 필요 없다 -
이 방식이 tests/conftest.py::synthetic_distorted_dataset_dir(실제 이미지
렌더링 + 검출)보다 훨씬 빠르고, ground-truth 오차를 정확히 통제할 수 있다.

이 파일은 pytest 테스트 파일이 아니다(test_*.py 아님) - pytest가 수집하지
않고, 다른 test_windshield_*.py들이 import해서 쓰는 순수 헬퍼 모듈이다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo

IMG_W, IMG_H = 1280, 800

# 6x5 평면 그리드(체스보드 코너 배치와 동일한 형태) - object_points로 사용.
_GRID_COLS, _GRID_ROWS = 6, 5
_SQUARE = 0.04

# 보드가 카메라 앞 다양한 위치/자세에 놓인 경우들 (rvec, tvec).
_POSES = [
    (np.array([0.05, -0.03, 0.0]), np.array([-0.15, -0.05, 0.6])),
    (np.array([0.02, 0.10, 0.05]), np.array([0.12, -0.08, 0.5])),
    (np.array([-0.08, 0.05, 0.0]), np.array([-0.05, 0.10, 0.7])),
    (np.array([0.0, -0.12, 0.02]), np.array([0.10, 0.05, 0.55])),
    (np.array([0.10, 0.02, -0.02]), np.array([-0.10, -0.10, 0.65])),
    (np.array([-0.05, -0.08, 0.03]), np.array([0.05, 0.12, 0.60])),
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.8])),
    (np.array([0.06, 0.06, -0.03]), np.array([-0.08, 0.08, 0.5])),
]


def _object_grid() -> np.ndarray:
    pts = []
    for r in range(_GRID_ROWS):
        for c in range(_GRID_COLS):
            pts.append([c * _SQUARE, r * _SQUARE, 0.0])
    return np.array(pts, dtype=np.float64).reshape(-1, 1, 3)


def default_camera_matrix_distortion() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[900.0, 0.0, IMG_W / 2], [0.0, 900.0, IMG_H / 2], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    return K, D


def default_camera_config() -> CameraConfig:
    return CameraConfig(width=IMG_W, height=IMG_H, sensor_name="windshield-test")


def build_synthetic_windshield_dataset(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    shear_k: float = 0.0,
) -> Dataset:
    """camera_matrix/distortion으로 noiseless 투영한 코너를 가진 프레임들을
    만든다.

    shear_k가 0이 아니면, 각 코너의 y좌표에 `shear_k * (x - cx)` 만큼 추가
    변위를 더한다 - 이미지 중심 기준 좌/우로 반대 방향 변위가 생기는 비강체
    (non-rigid) 패턴이다. 단일 rvec/tvec(강체 변환) + 카메라 모델로는 이
    패턴을 설명할 수 없으므로, solvePnP로 포즈를 다시 맞춰도 체계적인 잔차가
    남는다 - 실제 Windshield 굴절이 남기는 것과 같은 종류의 "설명 안 되는
    잔차"를 흉내낸다(균일한 픽셀 평행이동과 달리, PnP가 흡수해서 없애버릴 수
    없다).
    """
    obj = _object_grid()
    cx = float(camera_matrix[0, 2])
    frames: list[Frame] = []
    for i, (rvec, tvec) in enumerate(_POSES):
        frame_id = f"synth_{i:02d}"
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        pts = projected.reshape(-1, 2).astype(np.float64)
        if shear_k:
            pts[:, 1] += shear_k * (pts[:, 0] - cx)
        corners = pts.reshape(-1, 1, 2).astype(np.float32)
        center = pts.mean(axis=0)
        detection = DetectionResult(
            image_id=frame_id,
            success=True,
            corners=corners,
            object_points=obj.astype(np.float32),
            num_corners=corners.shape[0],
            board_area_ratio=0.2,
            board_center_px=(float(center[0]), float(center[1])),
        )
        frame = Frame(
            image_info=ImageInfo(image_id=frame_id, path=f"synthetic://{frame_id}", width=IMG_W, height=IMG_H),
            detection=detection,
            status=FrameStatus.DETECTED,
        )
        frames.append(frame)
    return Dataset(frames=frames)
