"""
camera_calibrator.calibration.models.common
==============================================

pinhole / extended_pinhole / fisheye 세 모델이 공통으로 쓰는 로직.

원래 pinhole.py 안에 있던 헬퍼들을 여기로 옮겼다. 세 모델이 "같은 구조"를
갖도록 강제하는 목적도 있다 - 여기 정의된 함수만 쓰면 세 모델의 결과가
자동으로 같은 방식(영역 구분 기준, 최소 프레임 조건 등)으로 계산된다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    RegionalError,
)  # noqa: F401 (RegionalError used in type hints)

# calibrateCamera류 함수가 최소한으로 요구하는 뷰(이미지) 개수.
# 이론상 3장부터 동작하지만, 안정적인 초점거리 추정을 위해 최소치로 둔다.
# Fisheye는 파라미터가 더 많아(k1~k4) 이론적으로는 더 많은 뷰가 필요하지만,
# "최소 조건"은 세 모델 동일하게 두고 부족하면 개별 함수가 실패로 반환하게 한다.
MIN_FRAMES_REQUIRED = 3
MIN_CORNERS_PER_FRAME = 4  # cv2.calibrateCamera 계열 최소 요구사항


def collect_calibration_inputs(
    dataset: Dataset,
) -> tuple[list[Frame], list[np.ndarray], list[np.ndarray]]:
    """검출 성공 + 활성화(enabled) + 최소 코너 수를 만족하는 프레임만 골라
    calibrateCamera 입력 형태(object_points 리스트, image_points 리스트)로 변환.
    """
    usable_frames = [
        f
        for f in dataset.enabled_frames
        if f.detection
        and f.detection.success
        and f.detection.num_corners >= MIN_CORNERS_PER_FRAME
    ]

    object_points = [f.detection.object_points for f in usable_frames]
    image_points = [f.detection.corners for f in usable_frames]
    return usable_frames, object_points, image_points


def infer_image_size(dataset: Dataset, camera_config: CameraConfig) -> tuple[int, int]:
    """CameraConfig에 해상도가 없으면 첫 번째 프레임에서 유추."""
    if camera_config.width and camera_config.height:
        return camera_config.width, camera_config.height

    for f in dataset.frames:
        if f.image_info.width and f.image_info.height:
            return f.image_info.width, f.image_info.height

    raise ValueError(
        "이미지 해상도를 알 수 없습니다. CameraConfig 또는 프레임에 width/height가 필요합니다."
    )


def classify_regions(cx: float, cy: float, w: int, h: int) -> list[str]:
    """보드 중심 좌표를 기준으로 이 프레임이 속하는 영역들을 반환.
    한 프레임이 여러 영역(예: left + top + corner)에 동시에 속할 수 있다.
    """
    x_third, y_third = w / 3, h / 3
    regions: list[str] = []

    horiz = "left" if cx < x_third else ("right" if cx > 2 * x_third else "center_x")
    vert = "top" if cy < y_third else ("bottom" if cy > 2 * y_third else "center_y")

    if horiz == "center_x" and vert == "center_y":
        regions.append("center")
    if horiz == "left":
        regions.append("left")
    if horiz == "right":
        regions.append("right")
    if vert == "top":
        regions.append("top")
    if vert == "bottom":
        regions.append("bottom")
    if horiz in ("left", "right") and vert in ("top", "bottom"):
        regions.append("corner")

    return regions


def compute_regional_error(
    frames: list[Frame],
    per_frame_error: dict[str, float],
    image_size: tuple[int, int],
) -> RegionalError:
    """설계 문서 4번 - Center/Left/Right/Top/Bottom/Corner RMS.

    세 모델(Pinhole/Extended/Fisheye) 모두 이 함수를 그대로 재사용하므로,
    영역 구분 기준이 모델마다 달라질 걱정 없이 공정하게 비교할 수 있다.
    """
    w, h = image_size
    buckets: dict[str, list[float]] = {
        "center": [], "left": [], "right": [], "top": [], "bottom": [], "corner": [],
    }

    for frame in frames:
        error = per_frame_error.get(frame.image_info.image_id)
        center = frame.detection.board_center_px if frame.detection else None
        if error is None or center is None:
            continue
        for region in classify_regions(center[0], center[1], w, h):
            buckets[region].append(error)

    def _avg(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return RegionalError(
        center=_avg(buckets["center"]),
        left=_avg(buckets["left"]),
        right=_avg(buckets["right"]),
        top=_avg(buckets["top"]),
        bottom=_avg(buckets["bottom"]),
        corner=_avg(buckets["corner"]),
    )


def fmt_optional(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def regional_edge_average(regional_error: RegionalError) -> float | None:
    """RegionalError에서 외곽(left/right/top/bottom/corner)만 평균낸 값.
    compare.py와 validation.py가 '외곽 오차'를 정의할 때 같은 기준을 쓰도록 공용화.
    """
    edge_vals = [
        v
        for v in (
            regional_error.left,
            regional_error.right,
            regional_error.top,
            regional_error.bottom,
            regional_error.corner,
        )
        if v is not None
    ]
    return float(np.mean(edge_vals)) if edge_vals else None


def undistort_image(
    image: np.ndarray,
    result: CalibrationResult,
    camera_config: CameraConfig,
    balance: float = 0.0,
) -> np.ndarray:
    """캘리브레이션 결과로 이미지를 보정(undistort). UI의 preview.py가 이 함수를
    호출한다 - 실제 OpenCV 왜곡 보정 로직은 여기(backend)에만 있고, UI는
    화면에 그리는 것만 담당한다 (백엔드/UI 분리 원칙, 설계 문서 16번).

    Fisheye는 cv2.undistort가 아니라 별도 remap 경로가 필요해 분기한다.
    balance: fisheye 전용, 0=최대 크롭(왜곡 없는 중심만) ~ 1=원본 화각 최대 보존.
    """
    if not result.success or result.camera_matrix is None or result.distortion is None:
        raise ValueError(f"실패한 CalibrationResult는 undistort할 수 없습니다: {result.error_message}")

    K, D = result.camera_matrix, result.distortion
    size = (camera_config.width, camera_config.height)

    if result.model_name == CameraModelType.FISHEYE:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, size, np.eye(3), balance=balance
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, size, cv2.CV_16SC2
        )
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return cv2.undistort(image, K, D)
