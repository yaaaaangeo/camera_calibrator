from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Phase B-2 안정화 - 지원하는 alignment model을 명확한 literal set으로
# 관리한다. 오타(예: "Translation", "afifne")가 조용히 AFFINE으로 처리되던
# 이전 버그를 막는다.
SUPPORTED_ALIGNMENT_METHODS = ("translation", "affine")

# Phase B-2 - 너무 큰 geometric transform은 정렬 자체가 실패했거나(다른
# scene을 잘못 짝지음) ECC가 이상한 지역해로 수렴했다는 신호일 수 있다.
# 이 비율(이미지 대각선 대비)을 넘는 translation은 WARNING으로 낮춘다 -
# 근거가 확실한 값이 아니라 실데이터로 재조정될 initial default다.
DEFAULT_MAX_REASONABLE_TRANSLATION_DIAGONAL_RATIO = 0.25


@dataclass
class AlignmentResult:
    aligned_reference: np.ndarray
    warp_matrix: np.ndarray
    score: float | None
    error_px: float | None
    status: str
    method: str
    warning_message: str | None = None
    # Phase B-1 안정화 - warp로 인해 생기는 인공적인 border 영역(원본
    # reference 이미지가 실제로 덮지 못하는 영역)을 표시하는 mask.
    # True/1 = 실제 reference 데이터가 있는 유효 픽셀, False/0 = warp
    # border(BORDER_REFLECT 등으로 채워진 인공 영역) - Reflection metric은
    # 이 mask 밖의 픽셀을 절대 포함하면 안 된다.
    valid_mask: Optional[np.ndarray] = None
    # Phase B-2 - Affine 사용 시 사람이 읽을 수 있는 진단(사용자 스펙).
    # Translation 모드에서는 rotation_deg=0/scale=1/shear_deg=0으로 고정.
    translation_x_px: Optional[float] = None
    translation_y_px: Optional[float] = None
    rotation_deg: Optional[float] = None
    scale: Optional[float] = None
    shear_deg: Optional[float] = None

    @property
    def validity(self):
        """Phase D-3 - 공통 `ResultValidity`로 읽는 read-only 뷰. `status`
        문자열 자체(직렬화 대상)는 바뀌지 않는다."""
        from calibration.result_validity import validity_from_legacy_status
        return validity_from_legacy_status(self.status)


def _decompose_affine(warp: np.ndarray) -> tuple[float, float, float, float, float]:
    """2x3 affine warp matrix를 (tx, ty, rotation_deg, scale, shear_deg)로
    분해한다(QR 유사 분해 - 표준적인 2D affine decomposition). Translation
    전용 warp(회전/scale/shear 없음)에도 그대로 적용 가능하다."""
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    a, b = float(warp[0, 0]), float(warp[0, 1])
    c, d = float(warp[1, 0]), float(warp[1, 1])
    scale_x = math.hypot(a, c)
    if scale_x <= 1e-9:
        return tx, ty, 0.0, 0.0, 0.0
    rotation = math.atan2(c, a)
    shear = math.atan2(a * b + c * d, scale_x * scale_x) if scale_x > 1e-9 else 0.0
    scale_y = (a * d - b * c) / scale_x
    return tx, ty, math.degrees(rotation), float((scale_x + abs(scale_y)) / 2.0), math.degrees(shear)


def align_reference_to_normal(
    normal_luma: np.ndarray,
    reference_luma: np.ndarray,
    *,
    method: str = "translation",
    enabled: bool = True,
) -> AlignmentResult:
    if normal_luma.shape != reference_luma.shape:
        raise ValueError("normal/reference images must have the same resolution")
    if not enabled:
        h, w = normal_luma.shape
        return AlignmentResult(
            reference_luma, np.eye(2, 3, dtype=np.float32), None, None, "not_run", "none",
            valid_mask=np.ones((h, w), dtype=bool),
        )

    # Phase B-2 안정화 - 지원하지 않는 method는 조용히 AFFINE으로 처리하지
    # 않고 명시적으로 거부한다.
    if method == "translation":
        motion = cv2.MOTION_TRANSLATION
    elif method == "affine":
        motion = cv2.MOTION_AFFINE
    else:
        raise ValueError(
            f"unsupported reflection alignment method: {method!r} "
            f"(supported: {SUPPORTED_ALIGNMENT_METHODS})"
        )

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(
            normal_luma.astype(np.float32),
            reference_luma.astype(np.float32),
            warp,
            motion,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6),
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error as e:
        return AlignmentResult(
            reference_luma,
            warp,
            None,
            None,
            "invalid",
            method,
            f"alignment failed: {e}",
        )

    h, w = normal_luma.shape
    aligned = cv2.warpAffine(
        reference_luma,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    # Phase B-1 안정화 - 실제 reference 데이터가 warp 후에도 덮고 있는
    # 영역만 valid로 표시한다. BORDER_CONSTANT(value=0)로 채운 뒤 0이 아닌
    # 영역만 valid로 삼는다 - aligned 이미지 자체는 여전히
    # BORDER_REFLECT(자연스러운 표시용)를 쓰지만, metric 계산에는 이 mask를
    # 함께 넘겨 border 영역을 제외한다.
    ones = np.ones((h, w), dtype=np.float32)
    valid_mask = cv2.warpAffine(
        ones, warp, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    ) >= 0.999

    error_px = float(np.linalg.norm(warp[:, 2])) if warp.shape == (2, 3) else None
    tx, ty, rotation_deg, scale, shear_deg = _decompose_affine(warp)

    diagonal = math.hypot(h, w)
    translation_mag = math.hypot(tx, ty)
    too_large_transform = (
        diagonal > 0 and translation_mag > DEFAULT_MAX_REASONABLE_TRANSLATION_DIAGONAL_RATIO * diagonal
    )

    if score < 0.70:
        status = "invalid"
    elif score < 0.90 or too_large_transform:
        # 점수가 충분히 높아도(>=0.90) transform 자체가 비정상적으로 크면
        # "good"으로 자동 격상하지 않는다 - 최소 warning으로 낮춰 사람이
        # 확인하게 한다.
        status = "warning"
    else:
        status = "good"

    warning_message = None
    if too_large_transform:
        warning_message = (
            f"alignment translation ({translation_mag:.1f}px) is unusually large relative to the "
            f"image diagonal ({diagonal:.1f}px) - this may indicate a mismatched pair or a bad ECC "
            "local optimum rather than a true windshield/camera offset."
        )

    return AlignmentResult(
        aligned, warp, float(score), error_px, status, method,
        warning_message=warning_message,
        valid_mask=valid_mask,
        translation_x_px=tx, translation_y_px=ty,
        rotation_deg=rotation_deg, scale=scale, shear_deg=shear_deg,
    )
