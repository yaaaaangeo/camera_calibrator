"""ArUco 사전·검출 파라미터·검출기 - 두 기능이 같은 설정으로 만든다.

Camera Intrinsic(ChArUco·AprilGrid 보드)과 CAMERA–LiDAR(구멍 4개 + 6x6 마커 보드)는
보드가 달라 검출 로직은 각자 갖지만, 마커를 찾는 OpenCV 설정은 여기 하나다.
"""

from __future__ import annotations

import cv2

aruco = cv2.aruco


def dictionary(name_or_id) -> "cv2.aruco.Dictionary":
    """'DICT_6X6_250' 또는 cv2.aruco.DICT_* 값 -> Dictionary."""
    dict_id = getattr(aruco, name_or_id, None) if isinstance(name_or_id, str) else name_or_id
    if dict_id is None:
        raise ValueError(f"알 수 없는 ArUco dictionary: {name_or_id}")
    return aruco.getPredefinedDictionary(dict_id)


def detector_params(small: bool = False) -> "cv2.aruco.DetectorParameters":
    """기본: 서브픽셀 코너 보정. `small`: 한 변 수십 px 짜리 작은 마커용.

    6x6 마커는 테두리 포함 8칸이라 25 px 이면 칸당 3 px - 기본 설정으로는 많이 놓친다.
    더 촘촘한 임계값 창, 칸당 더 많은 샘플, 느슨한 외곽선 근사로 대부분 되살린다.
    """
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if small:
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 53
        p.adaptiveThreshWinSizeStep = 4
        p.minMarkerPerimeterRate = 0.01
        p.perspectiveRemovePixelPerCell = 10
        p.perspectiveRemoveIgnoredMarginPerCell = 0.15
        p.polygonalApproxAccuracyRate = 0.05
    return p


_cache: dict = {}


def marker_detector(name_or_id, small: bool = False) -> "cv2.aruco.ArucoDetector":
    """사전 + 파라미터로 만든 ArucoDetector (같은 설정이면 한 번만 만든다)."""
    key = (name_or_id, small)
    if key not in _cache:
        _cache[key] = aruco.ArucoDetector(dictionary(name_or_id), detector_params(small))
    return _cache[key]
