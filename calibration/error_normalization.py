"""Resolution-independent representations of pixel reprojection errors.

Raw pixel errors remain the source measurements.  ``error / f_mean`` is the
resolution-independent quantity used for absolute quality decisions, while
the FHD-equivalent value is only a convenient display/export conversion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


FHD_REFERENCE_SIZE = (1920, 1080)
# The historical pixel thresholds were written for roughly-FHD cameras.  A
# nominal 950 px focal length maps them to normalized angular-image units
# without changing their meaning for existing FHD projects.
REFERENCE_FOCAL_PX = 950.0
FHD_ASPECT_RATIO_REL_TOLERANCE = 0.01
# No real camera has a pixel focal length this small at any resolution this
# app supports - below this we're almost certainly looking at a placeholder
# matrix (e.g. np.eye(3), used widely in tests as "don't care about K") or
# corrupted data, not a real intrinsic.  Below the floor we treat the matrix
# as invalid and fall back to raw px, same as None/non-finite/non-positive.
MIN_PLAUSIBLE_FOCAL_PX = 20.0


def mean_focal_length(camera_matrix: Any) -> float | None:
    """Return ``(fx + fy) / 2`` for a usable camera matrix, else ``None``."""
    if camera_matrix is None:
        return None
    try:
        matrix = np.asarray(camera_matrix, dtype=float)
        if matrix.shape != (3, 3):
            return None
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(fx) or not math.isfinite(fy) or fx <= 0.0 or fy <= 0.0:
        return None
    if fx < MIN_PLAUSIBLE_FOCAL_PX or fy < MIN_PLAUSIBLE_FOCAL_PX:
        return None
    return (fx + fy) / 2.0


def normalized_reprojection_error(error_px: float | None, camera_matrix: Any) -> float | None:
    """Compute ``error_px / f_mean``; invalid inputs produce ``None``."""
    focal = mean_focal_length(camera_matrix)
    if error_px is None or focal is None:
        return None
    try:
        error = float(error_px)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(error) or error < 0.0:
        return None
    return error / focal


def reference_equivalent_error(
    error_px: float | None,
    camera_matrix: Any,
    reference_focal_px: float = REFERENCE_FOCAL_PX,
) -> float | None:
    """Map a normalized error to the nominal reference-focal pixel scale."""
    normalized = normalized_reprojection_error(error_px, camera_matrix)
    if normalized is None or not math.isfinite(reference_focal_px) or reference_focal_px <= 0.0:
        return None
    return normalized * reference_focal_px


def fhd_equivalent_error(
    error_px: float | None,
    image_size: tuple[int, int] | None,
    *,
    aspect_ratio_tolerance: float = FHD_ASPECT_RATIO_REL_TOLERANCE,
) -> float | None:
    """Scale a pixel error to 1920x1080 when the aspect ratio is compatible.

    The average of horizontal and vertical scale factors is used as requested.
    Non-16:9 inputs deliberately return ``None`` instead of implying a crop or
    stretch that was not part of calibration.
    """
    if error_px is None or image_size is None:
        return None
    try:
        error = float(error_px)
        width, height = float(image_size[0]), float(image_size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if (
        not all(math.isfinite(v) for v in (error, width, height))
        or error < 0.0
        or width <= 0.0
        or height <= 0.0
    ):
        return None
    source_ratio = width / height
    target_ratio = FHD_REFERENCE_SIZE[0] / FHD_REFERENCE_SIZE[1]
    if abs(source_ratio / target_ratio - 1.0) > aspect_ratio_tolerance:
        return None
    scale = ((FHD_REFERENCE_SIZE[0] / width) + (FHD_REFERENCE_SIZE[1] / height)) / 2.0
    return error * scale
