"""
tests/test_fisheye_import_safety.py
========================================

실제 사용자가 보고한 크래시의 회귀 테스트:

    AttributeError: module 'cv2.fisheye' has no attribute 'CALIB_RECOMPUTE_EXTRINSIC'

원인: calibration/models/fisheye.py가 모듈 최상단(import 시점)에서
cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC 같은 플래그를 직접 참조했다.
OpenCV 빌드/버전에 따라 이 플래그가 없으면 fisheye 모델을 쓰지도 않는
사용자조차 앱 전체가 뜨지 못했다 (`python -m app.main`이 즉시 죽음).

수정: cv2.fisheye 플래그 조회를 함수 내부로 옮기고, getattr(..., default=0)로
없는 플래그는 조용히 건너뛰게 했다. 이 테스트는 그 플래그가 없는 OpenCV
환경을 흉내 내서, import와 실제 캘리브레이션이 여전히 동작하는지 확인한다.
"""

from __future__ import annotations

import importlib
import sys

import cv2
import numpy as np
import pytest


@pytest.fixture
def fisheye_missing_flags(request):
    """cv2.fisheye에서 특정 플래그들이 없는 것처럼 흉내 낸다. 테스트가 끝나면
    원래 cv2.fisheye로 복구하고 calibration.models.fisheye 모듈도 재로드한다.
    """
    missing_names = request.param
    real_fisheye = cv2.fisheye

    class _FakeFisheyeModule:
        def __getattr__(self, name):
            if name in missing_names:
                raise AttributeError(name)
            return getattr(real_fisheye, name)

    cv2.fisheye = _FakeFisheyeModule()
    sys.modules.pop("calibration.models.fisheye", None)

    yield real_fisheye

    cv2.fisheye = real_fisheye
    sys.modules.pop("calibration.models.fisheye", None)
    import calibration.models.fisheye  # noqa: F401


@pytest.mark.parametrize(
    "fisheye_missing_flags",
    [
        ("CALIB_RECOMPUTE_EXTRINSIC",),  # 실제 사용자가 겪은 정확한 케이스
        ("CALIB_RECOMPUTE_EXTRINSIC", "CALIB_FIX_SKEW"),
        ("CALIB_CHECK_COND",),
        ("CALIB_USE_INTRINSIC_GUESS",),
    ],
    indirect=True,
)
def test_module_imports_without_crashing_when_flags_missing(fisheye_missing_flags):
    """이게 이 버그의 핵심 - import 자체가 죽으면 안 된다."""
    import calibration.models.fisheye as fm
    importlib.reload(fm)


@pytest.mark.parametrize(
    "fisheye_missing_flags",
    [("CALIB_RECOMPUTE_EXTRINSIC", "CALIB_FIX_SKEW")],
    indirect=True,
)
def test_calibration_still_succeeds_with_missing_flags(fisheye_missing_flags):
    """import만 살아있는 게 아니라, 실제 캘리브레이션 계산도 여전히 동작해야
    의미가 있다.
    """
    real_fisheye = fisheye_missing_flags
    import calibration.models.fisheye as fm
    importlib.reload(fm)

    from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict)
    pts3d = board.getChessboardCorners().astype(np.float32)

    true_K = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])
    true_D = np.array([0.05, 0.01, -0.01, 0.002])

    rng = np.random.default_rng(0)
    frames = []
    for i in range(15):
        rvec = (rng.random(3) - 0.5) * 0.8
        tvec = np.array([(rng.random() - 0.5) * 0.3, (rng.random() - 0.5) * 0.3, 0.4 + rng.random() * 0.3])
        proj, _ = real_fisheye.projectPoints(
            pts3d.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, true_K, true_D
        )
        proj = proj.reshape(-1, 2)
        if (proj < 0).any() or (proj[:, 0] > 1280).any() or (proj[:, 1] > 720).any():
            continue
        info = ImageInfo(image_id=f"f{i}", path="-", width=1280, height=720)
        det = DetectionResult(
            image_id=f"f{i}", success=True, corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), num_corners=pts3d.shape[0],
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    assert len(frames) >= 8
    dataset = Dataset(frames=frames)
    camera_config = CameraConfig(width=1280, height=720)

    result = fm.calibrate_fisheye(dataset, camera_config)
    assert result.success, f"플래그 일부가 없어도 캘리브레이션은 성공해야 함: {result.error_message}"


def test_fisheye_base_flags_only_includes_available_attrs():
    """_fisheye_base_flags()가 정상 환경에서는 여전히 정상 동작하는지."""
    import calibration.models.fisheye as fm
    flags = fm._fisheye_base_flags()
    assert isinstance(flags, int)
    assert flags != 0, "정상 OpenCV 환경에서는 플래그가 최소 하나는 있어야 함"
