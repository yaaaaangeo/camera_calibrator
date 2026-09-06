"""
calibration.windshield.ghost.edge_detector
==============================================================

STEP 8A - Edge/Line Ghost Evaluation.

고대비 직선 target(도로 차선 같은 target)의 1D intensity profile ->
gradient profile -> primary gradient peak(진짜 edge) + secondary
gradient peak(ghost로 인한 약하고 shift된 edge)을 검출한다(사용자
스펙 19-21번). Blur(broadened single peak)와 Ghost(two distinct
peaks)를 구분하는 것이 이 모듈의 핵심 역할이다(사용자 스펙 32-33번).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from calibration.windshield.ghost.config import (
    DEFAULT_EDGE_CANNY_HIGH,
    DEFAULT_EDGE_CANNY_LOW,
    DEFAULT_EDGE_HOUGH_MAX_LINE_GAP_PX,
    DEFAULT_EDGE_HOUGH_MIN_LINE_LENGTH_PX,
    DEFAULT_EDGE_HOUGH_THRESHOLD,
    DEFAULT_EDGE_MAX_PROFILE_SAMPLES,
    DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX,
    DEFAULT_EDGE_MIN_GRADIENT,
    DEFAULT_EDGE_MIN_SECONDARY_RATIO,
    DEFAULT_EDGE_ORIENTATION_TOLERANCE_DEG,
    DEFAULT_EDGE_PROFILE_HALF_LENGTH_PX,
)
from calibration.windshield.ghost.types import GhostEvaluationConfig

# Primary peak 바로 주변(같은 broadened peak에 속할 가능성이 높은 구간)은
# secondary peak 탐색에서 제외한다 - 이게 없으면 blur로 넓어진 하나의
# peak의 옆 기슭을 "ghost"로 오검출하게 된다.
_PRIMARY_EXCLUSION_RADIUS_PX = 3


@dataclass
class EdgeGhostDetection:
    primary_index: int
    primary_value: float
    secondary_index: Optional[int] = None
    secondary_value: Optional[float] = None
    offset_px: Optional[float] = None
    strength_ratio: Optional[float] = None
    detected: bool = False


def compute_gradient_profile(intensity_profile: np.ndarray) -> np.ndarray:
    """1D intensity profile의 gradient magnitude profile을 계산한다."""
    profile = np.asarray(intensity_profile, dtype=np.float64)
    return np.abs(np.gradient(profile))


def detect_edge_ghost(
    intensity_profile: np.ndarray,
    *,
    min_gradient: float = DEFAULT_EDGE_MIN_GRADIENT,
    max_search_radius_px: float = DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX,
    min_secondary_ratio: float = DEFAULT_EDGE_MIN_SECONDARY_RATIO,
) -> EdgeGhostDetection:
    """1D intensity profile에서 primary edge와 (있다면) secondary(ghost)
    edge를 찾는다.

    Primary = 전체 gradient profile의 global maximum(가장 강한 진짜
    edge). Secondary = `[primary-radius, primary+radius]` 구간에서
    primary 바로 옆(`_PRIMARY_EXCLUSION_RADIUS_PX`) 을 제외한 나머지
    중 가장 큰 local maximum. Secondary/Primary 비율이
    `min_secondary_ratio` 미만이면 noise로 보고 ghost 없음으로 처리한다.
    """
    grad = compute_gradient_profile(intensity_profile)
    n = grad.shape[0]

    primary_idx = int(np.argmax(grad))
    primary_val = float(grad[primary_idx])

    if primary_val < min_gradient:
        return EdgeGhostDetection(primary_index=primary_idx, primary_value=primary_val, detected=False)

    radius = int(round(max_search_radius_px))
    lo = max(0, primary_idx - radius)
    hi = min(n, primary_idx + radius + 1)

    best_idx = None
    best_val = -1.0
    for idx in range(lo, hi):
        if abs(idx - primary_idx) <= _PRIMARY_EXCLUSION_RADIUS_PX:
            continue
        # local maximum 조건: 이웃보다 크거나 같아야 진짜 peak다(평평한
        # noise floor를 peak로 잘못 세는 것을 방지).
        left = grad[idx - 1] if idx - 1 >= 0 else -np.inf
        right = grad[idx + 1] if idx + 1 < n else -np.inf
        if grad[idx] >= left and grad[idx] >= right and grad[idx] > best_val:
            best_val = float(grad[idx])
            best_idx = idx

    if best_idx is None:
        return EdgeGhostDetection(primary_index=primary_idx, primary_value=primary_val, detected=False)

    ratio = best_val / (primary_val + 1e-9)
    if ratio < min_secondary_ratio:
        return EdgeGhostDetection(primary_index=primary_idx, primary_value=primary_val, detected=False)

    return EdgeGhostDetection(
        primary_index=primary_idx,
        primary_value=primary_val,
        secondary_index=best_idx,
        secondary_value=best_val,
        offset_px=float(best_idx - primary_idx),
        strength_ratio=float(ratio),
        detected=True,
    )


# ---------------------------------------------------------------------------
# STEP 8 stabilization 2번 - Image -> 1D edge profile 추출.
#
# 사용자가 1D numpy profile을 직접 만들어 UI에 넣게 하지 않는다 - 이미지
# 입력만으로 동작해야 한다. 범용 line detector를 만들지 않고, 고대비 직선
# calibration target을 안정적으로 다루는 첫 버전만 구현한다:
#
#   Luminance -> Canny -> HoughLinesP(dominant edge) -> edge normal 방향
#   -> normal을 따라 여러 지점에서 bilinear 1D profile sampling
# ---------------------------------------------------------------------------

def _to_gray_uint8(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        gray = image_bgr.astype(np.float32)
    else:
        gray = cv2.cvtColor(image_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if image_bgr.dtype != np.uint8 and gray.max() <= 1.0 + 1e-6:
        gray = gray * 255.0
    return np.clip(gray, 0, 255).astype(np.uint8)


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _matches_edge_axis(angle_deg: float, edge_axis: str, tolerance_deg: float) -> bool:
    """`edge_axis`가 "vertical"/"horizontal"이면 검출된 line의 방향이 그
    축과 충분히 가까운지 확인한다. "auto"는 방향에 상관없이 항상 통과."""
    if edge_axis == "auto":
        return True
    a = abs(angle_deg) % 180.0
    if edge_axis == "vertical":
        return abs(a - 90.0) <= tolerance_deg
    if edge_axis == "horizontal":
        return a <= tolerance_deg or a >= 180.0 - tolerance_deg
    return True


def _pick_dominant_line(
    lines: np.ndarray, edge_axis: str, tolerance_deg: float,
) -> Optional[tuple[float, float, float, float]]:
    """검출된 Hough 선분들 중 `edge_axis` 방향 제약을 만족하는 가장 긴
    선분을 dominant edge로 고른다."""
    best = None
    best_len = -1.0
    for line in lines:
        # cv2.HoughLinesP()의 반환 shape은 OpenCV 버전에 따라 (N,1,4) 또는
        # (N,4)일 수 있다(OpenCV 5 CI에서 (N,4)로 확인됨) - line[0]이 4개
        # 좌표의 하위 배열이 아니라 scalar가 되는 경우 `float(v) for v in
        # line[0]`이 `TypeError: 'numpy.int32' object is not iterable`을
        # 낸다. reshape(-1)로 두 shape 모두 안전하게 4개 좌표로 펼친다.
        coords = np.asarray(line).reshape(-1)
        if coords.size != 4:
            continue
        x1, y1, x2, y2 = map(float, coords)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-6:
            continue
        angle = _line_orientation_deg(x1, y1, x2, y2)
        if not _matches_edge_axis(angle, edge_axis, tolerance_deg):
            continue
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)
    return best


def _sample_profile_along_normal(
    gray: np.ndarray, cx: float, cy: float, normal: tuple[float, float], half_length: float,
) -> np.ndarray:
    """`(cx,cy)`를 중심으로 `normal` 방향을 따라 `[-half_length, +half_length]`
    구간을 1px 간격으로 bilinear sampling한 1D profile을 만든다
    (`scipy.ndimage.map_coordinates`, order=1 - 새 dependency 없이 정확한
    bilinear 보간)."""
    nx, ny = normal
    ts = np.arange(-half_length, half_length + 1.0, 1.0)
    xs = cx + ts * nx
    ys = cy + ts * ny
    return map_coordinates(gray.astype(np.float64), [ys, xs], order=1, mode="nearest")


def extract_edge_profiles(
    image_bgr: np.ndarray,
    config: Optional["GhostEvaluationConfig"] = None,
) -> list[np.ndarray]:
    """이미지에서 고대비 직선(dominant edge)을 찾아 그 edge normal 방향의
    1D intensity profile 여러 개를 만든다(STEP 8 stabilization 2-B번).

    Pipeline: Luminance -> Canny -> HoughLinesP -> dominant line 선택
    (`edge_axis` 방향 제약 적용) -> line을 따라 균등 간격으로 여러 지점을
    골라 각 지점에서 normal 방향 1D profile 추출.

    Dominant edge를 찾지 못하면 빈 리스트를 반환한다 - 호출자
    (`evaluate_ghost_edge_target`)는 이미 빈 리스트를 "profile 없음"으로
    안전하게 처리한다.
    """
    cfg = config
    edge_axis = getattr(cfg, "edge_axis", "vertical") if cfg is not None else "vertical"
    half_length = DEFAULT_EDGE_PROFILE_HALF_LENGTH_PX
    max_samples = DEFAULT_EDGE_MAX_PROFILE_SAMPLES

    gray = _to_gray_uint8(image_bgr)
    edges = cv2.Canny(gray, DEFAULT_EDGE_CANNY_LOW, DEFAULT_EDGE_CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=DEFAULT_EDGE_HOUGH_THRESHOLD,
        minLineLength=DEFAULT_EDGE_HOUGH_MIN_LINE_LENGTH_PX,
        maxLineGap=DEFAULT_EDGE_HOUGH_MAX_LINE_GAP_PX,
    )
    if lines is None or len(lines) == 0:
        return []

    dominant = _pick_dominant_line(lines, edge_axis, DEFAULT_EDGE_ORIENTATION_TOLERANCE_DEG)
    if dominant is None:
        return []
    x1, y1, x2, y2 = dominant

    length = math.hypot(x2 - x1, y2 - y1)
    dir_x, dir_y = (x2 - x1) / length, (y2 - y1) / length
    normal = (-dir_y, dir_x)

    num_samples = int(np.clip(round(length / 20.0), 3, max_samples))
    h, w = gray.shape[:2]
    profiles: list[np.ndarray] = []
    for t in np.linspace(0.15, 0.85, num_samples):
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        profiles.append(_sample_profile_along_normal(gray, cx, cy, normal, half_length))

    return profiles
