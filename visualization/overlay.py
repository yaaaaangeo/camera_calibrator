"""
visualization/overlay.py

Renders the single most useful diagnostic image for this tool: the actual
camera frame with the LiDAR edge points (from M2's edge_alignment) drawn on
top, color-coded per-point by how far each one landed from the nearest
image edge, relative to the sensor's own noise floor. This is the
"projected LiDAR ↔ image edge" picture called out repeatedly in the design
notes as the most immediately legible way to show someone whether a
calibration is good.

Per-point color uses the SAME GOOD/WARNING/BAD multiplier scheme as M2's
overall classification (quality.noise_floor.classify), so a person looking
at the image and a person looking at the M2 score are seeing the same
underlying judgment, just at different granularity (per-point vs
aggregate).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

from quality.noise_floor import classify, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER


_COLOR_BGR = {
    # OpenCV uses BGR, not RGB
    "GOOD": (79, 185, 63),      # matches report --good (#3FB950) reversed to BGR-ish
    "WARNING": (34, 153, 210),  # matches report --warning (#D29922)
    "BAD": (81, 81, 248),       # matches report --bad (#F85149)
}

_POINT_RADIUS = 2
_POINT_THICKNESS = -1  # filled


def render_overlay(
    image: np.ndarray,
    edge_pixels: np.ndarray,
    errors_px: np.ndarray,
    floor_px: float,
    good_mult: float = M2_GOOD_MULTIPLIER,
    warning_mult: float = M2_WARNING_MULTIPLIER,
    point_radius: int = _POINT_RADIUS,
    draw_edge_map: bool = True,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """
    Draw edge_pixels on top of `image`, colored GOOD/WARNING/BAD per-point
    using the same floor(Z)-relative thresholds M2 uses for its aggregate
    classification. Returns a new BGR image (does not mutate the input).

    If draw_edge_map is True, the detected image edges (what the LiDAR
    points are being compared against) are drawn faintly underneath in a
    dim gray, so the person can see WHY a point is colored the way it is
    -- how far it actually sits from the nearest edge -- not just the dot.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if draw_edge_map:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, canny_low, canny_high)
        edge_overlay = np.zeros_like(canvas)
        edge_overlay[edges > 0] = (90, 90, 90)
        canvas = cv2.addWeighted(canvas, 1.0, edge_overlay, 0.9, 0)

    for (u, v), err in zip(edge_pixels, errors_px):
        classification = classify(float(err), floor_px, good_mult, warning_mult)
        color = _COLOR_BGR.get(classification, (170, 170, 170))
        cv2.circle(canvas, (int(round(u)), int(round(v))), point_radius, color, _POINT_THICKNESS,
                   lineType=cv2.LINE_AA)

    return canvas


def render_overlay_from_result(image: np.ndarray, edge_alignment_result, **kwargs) -> Optional[np.ndarray]:
    """
    Convenience wrapper taking an EdgeAlignmentResult directly (as returned
    by evaluation.edge_alignment.evaluate_edge_alignment). Returns None if
    the result FAILed (no per-point data to draw), so callers can skip the
    visualization gracefully rather than crashing on missing arrays.
    """
    if edge_alignment_result.classification == "FAIL" or edge_alignment_result.edge_point_pixels is None:
        return None
    return render_overlay(
        image,
        edge_alignment_result.edge_point_pixels,
        edge_alignment_result.edge_point_errors_px,
        edge_alignment_result.floor_px,
        **kwargs,
    )


def encode_png(image_bgr: np.ndarray) -> bytes:
    """Encode a BGR image array to PNG bytes (for embedding or writing to disk)."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode overlay image to PNG.")
    return buf.tobytes()


def save_overlay_png(image_bgr: np.ndarray, path: str) -> None:
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write overlay image to {path!r}.")
