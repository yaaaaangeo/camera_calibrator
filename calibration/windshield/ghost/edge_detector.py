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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibration.windshield.ghost.config import (
    DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX,
    DEFAULT_EDGE_MIN_GRADIENT,
    DEFAULT_EDGE_MIN_SECONDARY_RATIO,
)

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
