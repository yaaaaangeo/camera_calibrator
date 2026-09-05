from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlignmentResult:
    aligned_reference: np.ndarray
    warp_matrix: np.ndarray
    score: float | None
    error_px: float | None
    status: str
    method: str
    warning_message: str | None = None


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
        return AlignmentResult(reference_luma, np.eye(2, 3, dtype=np.float32), None, None, "not_run", "none")

    motion = cv2.MOTION_TRANSLATION if method == "translation" else cv2.MOTION_AFFINE
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
    error_px = float(np.linalg.norm(warp[:, 2])) if warp.shape == (2, 3) else None
    if score >= 0.90:
        status = "good"
    elif score >= 0.70:
        status = "warning"
    else:
        status = "invalid"
    return AlignmentResult(aligned, warp, float(score), error_px, status, method)
