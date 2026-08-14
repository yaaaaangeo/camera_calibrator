"""
tests/test_fisheye_opencv5_compat.py
=========================================

실제 사용자가 OpenCV 5.0.0(opencv-contrib-python==5.0.0.93)에서 보고한 크래시의
회귀 테스트:

    [Fisheye] cv2.fisheye.calibrate 실패 (완화된 조건으로 재시도도 실패):
    OpenCV(5.0.0) .../fisheye.cpp:64: error: (-215:Assertion failed)
    objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3
    in function 'calibrate'

pip으로 opencv-contrib-python==5.0.0.93을 직접 설치해 사용자와 완전히 동일한
환경에서 재현하고 고쳤다. 원인 두 가지:

1. OpenCV 5.0부터 cv2.fisheye.CALIB_* 플래그들이 최상위 cv2.CALIB_*로
   옮겨갔다 (cv2.fisheye 네임스페이스엔 더 이상 없음) - _fisheye_flag()가
   cv2.fisheye에서만 찾다가 전부 실패해서 flags=0이 되고 있었다.
2. cv2.fisheye.calibrate()는 (N,1,3)/(N,1,2) shape(cv2.calibrateCamera가
   쓰는 관례)을 안 받고 (1,N,3)/(1,N,2)를 요구한다 - OpenCV 4.13.0에서는
   그냥 넘어가던 게 5.0.0부터 엄격해진 것으로 보인다.

이 테스트는 CI가 어떤 OpenCV 버전을 쓰든(4.x/5.x) 통과해야 한다 - 두 버전
모두에서 이 shape/flag 조합이 정상 동작함을 직접 확인했기 때문이다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.models.fisheye import _fisheye_base_flags, _fisheye_flag, _to_fisheye_points, calibrate_fisheye
from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo

pytestmark = pytest.mark.slow


def test_fisheye_flag_falls_back_to_top_level_cv2():
    """cv2.fisheye에 없으면 최상위 cv2에서 찾아야 한다 (OpenCV 5.x 대응)."""
    value = _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC")
    assert value != 0, (
        "CALIB_RECOMPUTE_EXTRINSIC을 cv2.fisheye와 최상위 cv2 어디에서도 못 찾음 - "
        "설치된 OpenCV에 이 플래그 자체가 없거나 fallback 로직이 깨짐"
    )


def test_fisheye_base_flags_nonzero():
    """세 플래그가 최소 하나는 잡혀야 한다 - 전부 0이면 OpenCV 5.x에서
    flags=0이 되던 원래 버그가 재발한 것.
    """
    flags = _fisheye_base_flags()
    assert flags != 0


def test_to_fisheye_points_reshapes_to_1_n_3():
    """cv2.fisheye.calibrate()가 요구하는 (1,N,3)/(1,N,2) shape으로
    변환되는지 직접 확인.
    """
    obj = [np.zeros((24, 1, 3), dtype=np.float32)]
    img = [np.zeros((24, 1, 2), dtype=np.float32)]

    obj64, img64 = _to_fisheye_points(obj, img)

    assert obj64[0].shape == (1, 24, 3)
    assert img64[0].shape == (1, 24, 2)
    assert obj64[0].dtype == np.float64
    assert img64[0].dtype == np.float64


def _build_synthetic_fisheye_frames(n_frames=15, seed=0):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((11, 8), 0.02, 0.015, aruco_dict)
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    true_K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    true_D = np.array([0.05, 0.01, -0.01, 0.002])

    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_frames):
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.2, (rng.random() - 0.5) * 0.2, 0.3 + rng.random() * 0.2])
        proj, _ = cv2.fisheye.projectPoints(
            pts3d.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, true_K, true_D
        )
        proj = proj.reshape(-1, 2)
        if (proj < 0).any() or (proj[:, 0] > 640).any() or (proj[:, 1] > 480).any():
            continue
        info = ImageInfo(image_id=f"f{i}", path="-", width=640, height=480)
        det = DetectionResult(
            image_id=f"f{i}", success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=n_corners,
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))
    return frames


def test_calibrate_fisheye_end_to_end_matches_reported_board_config():
    """사용자가 실제로 겪은 정확한 보드 설정(11x8, DICT_4X4_100, 640x480)으로
    전체 calibrate_fisheye()가 성공해야 한다 - 이게 이 버그의 핵심 회귀 테스트.
    """
    frames = _build_synthetic_fisheye_frames()
    assert len(frames) >= 8, "합성 프레임이 너무 적게 생성됨 - 테스트 파라미터 조정 필요"

    dataset = Dataset(frames=frames)
    camera_config = CameraConfig(width=640, height=480)

    result = calibrate_fisheye(dataset, camera_config)

    assert result.success, f"Fisheye 캘리브레이션이 실패함: {result.error_message}"
    assert result.camera_matrix is not None
    assert result.rms_error is not None
    assert len(result.per_frame_error) == len(frames)
    assert result.radial_profile is not None
    assert len(result.radial_profile.bins) > 0
