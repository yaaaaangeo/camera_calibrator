"""
calibration.windshield.ghost.synthetic
==============================================================

STEP 8A/8B - Synthetic Ghost Ground Truth 생성기.

`calibration.windshield.reflection_suppression.synthetic`(Exterior +
Interior 두 개의 서로 다른 scene을 합성)과는 **완전히 독립된 모듈**이다
(사용자 스펙 27-29번, "Reflection synthetic ≠ Ghost synthetic"). 여기서는
항상 하나의 clean exterior 이미지에서 자기 자신의 shifted/warped copy를
만들어 합성한다:

    ghost = warp(clean, dx, dy)
    observed = clip(clean + alpha * ghost, 0, 1)

`dx`, `dy`, `alpha`는 상수(constant offset)이거나, 픽셀 위치에 따라
달라지는 필드(spatially-varying, `dx(u,v)`/`dy(u,v)`)일 수 있다(사용자
스펙 29-30번, D 테스트: Left/Center/Right offset이 다른 경우).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class SyntheticGhostSample:
    clean: np.ndarray
    ghost: np.ndarray
    observed: np.ndarray
    dx: float | np.ndarray
    dy: float | np.ndarray
    alpha: float | np.ndarray


def warp_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """이미지 전체를 (dx, dy)만큼 constant translation으로 이동시킨다.

    Ghost 정의(사용자 스펙 3번, `I ~= T(x,y) + alpha*T(x-dx,y-dy)`)에 맞춰,
    출력의 (x,y) 위치에는 입력의 (x-dx, y-dy) 값이 오도록 만든다.
    """
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    h, w = image.shape[:2]
    return cv2.warpAffine(image.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp_shift_field(image: np.ndarray, dx_field: np.ndarray, dy_field: np.ndarray) -> np.ndarray:
    """픽셀별로 다른 (dx(x,y), dy(x,y))를 갖는 spatially-varying shift를
    적용한다.

    출력 (x,y)에 입력의 (x-dx(x,y), y-dy(x,y))가 오도록
    `map_x = xx - dx_field`, `map_y = yy - dy_field`로 `cv2.remap`을 쓴다.
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xx - dx_field.astype(np.float32)).astype(np.float32)
    map_y = (yy - dy_field.astype(np.float32)).astype(np.float32)
    return cv2.remap(
        image.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def make_constant_offset_ghost_sample(
    clean: np.ndarray,
    *,
    dx: float = 4.0,
    dy: float = 2.0,
    alpha: float = 0.15,
) -> SyntheticGhostSample:
    """상수 offset/strength ghost 합성 샘플을 만든다(사용자 스펙 27번의
    기본 예시: `dx=4, dy=2, alpha=0.15`)."""
    clean_f = clean.astype(np.float32)
    ghost = warp_shift(clean_f, dx, dy)
    observed = np.clip(clean_f + alpha * ghost, 0.0, 1.0 if clean_f.max() <= 1.0 + 1e-6 else 255.0)
    return SyntheticGhostSample(clean=clean_f, ghost=ghost, observed=observed, dx=dx, dy=dy, alpha=alpha)


def make_spatially_varying_ghost_sample(
    clean: np.ndarray,
    *,
    dx_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alpha: float = 0.15,
) -> SyntheticGhostSample:
    """픽셀 위치에 따라 달라지는 `dx_fn(x,y)`/`dy_fn(x,y)` 필드로 ghost를
    합성한다(사용자 스펙 30번, Test D: Left/Center/Right가 다른 offset을
    갖는 경우를 검증하기 위함).

    `dx_fn`/`dy_fn`은 `np.meshgrid`로 만든 (H,W) 크기의 x,y 좌표 배열을
    받아 같은 크기의 offset 배열을 반환해야 한다.
    """
    clean_f = clean.astype(np.float32)
    h, w = clean_f.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx_field = np.asarray(dx_fn(xx, yy), dtype=np.float32)
    dy_field = np.asarray(dy_fn(xx, yy), dtype=np.float32)

    ghost = warp_shift_field(clean_f, dx_field, dy_field)
    high = 1.0 if clean_f.max() <= 1.0 + 1e-6 else 255.0
    observed = np.clip(clean_f + alpha * ghost, 0.0, high)
    return SyntheticGhostSample(clean=clean_f, ghost=ghost, observed=observed, dx=dx_field, dy=dy_field, alpha=alpha)


def make_left_center_right_offset_fields(
    image_width: int,
    *,
    left_dx: float,
    center_dx: float,
    right_dx: float,
    dy: float = 0.0,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """왼쪽/가운데/오른쪽 영역에서 서로 다른 dx를 갖는 spatially-varying
    ghost 필드를 만들기 위한 편의 함수(사용자 스펙 30번 D 테스트 전용
    helper). x 좌표에 대해 3구간 선형보간한 dx와, 상수 dy를 돌려준다."""

    def dx_fn(xx: np.ndarray, _yy: np.ndarray) -> np.ndarray:
        nx = xx / max(image_width - 1, 1)
        # 0 -> left_dx, 0.5 -> center_dx, 1 -> right_dx의 구간 선형보간.
        left_half = nx <= 0.5
        out = np.empty_like(nx)
        out[left_half] = left_dx + (center_dx - left_dx) * (nx[left_half] / 0.5)
        out[~left_half] = center_dx + (right_dx - center_dx) * ((nx[~left_half] - 0.5) / 0.5)
        return out

    def dy_fn(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return np.full_like(xx, dy)

    return dx_fn, dy_fn
