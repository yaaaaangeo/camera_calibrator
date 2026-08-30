"""
camera_calibrator.camera_lidar.guided_roi
=============================================

GUIDED AUTO ROI: uses a Targetless (direct_visual_lidar_calibration) prior
plus the camera's pose-inferred board circle centers to predict where the
calibration board should be in the LiDAR frame, then builds a
spatially-constrained AABB search region around that prediction --
narrowing the LiDAR search space before FAST-Calib's own multi-plane
circle detection runs there (camera_lidar.lidar_detector.detect_lidar_target_auto's
search_roi parameter).

Pure geometry/helper module: no LiDAR circle/plane detection happens here.
The Targetless prior's ONLY influence is on which points get handed to the
existing FAST-Calib detector -- never on the final correspondence/solve
(see camera_lidar/types.py's TargetlessPrior docstring).

1st-implementation scope: AABB only, no oriented (board-normal-aligned)
box -- the coarse rotation itself carries uncertainty, so an OBB tight
enough to be useful risks cropping out the real target.
"""

from __future__ import annotations

import numpy as np

from camera_lidar.types import GuidedROIConfig, ROIConfig
from geometry.transform import transform_points


def predict_circle_centers_lidar(
    camera_circle_centers: np.ndarray,
    T_lidar_from_camera: np.ndarray,
) -> np.ndarray:
    """Map the camera's (4,3) pose-inferred circle centers into the LiDAR
    frame via the Targetless prior. All 4 pose-inferred centers are used
    here regardless of how many markers the camera actually, independently
    detected -- this is a *search region prediction*, not a correspondence
    (see camera_lidar/camera_detector.py's module docstring on
    detected_ids vs. pose-inferred circle_centers)."""
    camera_circle_centers = np.asarray(camera_circle_centers, dtype=np.float64)
    T_lidar_from_camera = np.asarray(T_lidar_from_camera, dtype=np.float64)

    if camera_circle_centers.shape != (4, 3):
        raise ValueError(f"camera_circle_centers must be (4,3), got {camera_circle_centers.shape}")
    if T_lidar_from_camera.shape != (4, 4):
        raise ValueError(f"T_lidar_from_camera must be (4,4), got {T_lidar_from_camera.shape}")
    if not np.all(np.isfinite(camera_circle_centers)):
        raise ValueError("camera_circle_centers contains non-finite values")
    if not np.all(np.isfinite(T_lidar_from_camera)):
        raise ValueError("T_lidar_from_camera contains non-finite values")

    return transform_points(T_lidar_from_camera, camera_circle_centers)


def compute_guided_base_margin(
    predicted_circle_centers_lidar: np.ndarray,
    config: GuidedROIConfig,
) -> tuple[float, float]:
    """Adaptive ROI margin: Targetless translation uncertainty + a
    rotation-uncertainty term that grows with range (a fixed angular error
    becomes a larger positional error the farther the board is) + a fixed
    safety margin, clipped into [min_margin_m, max_margin_m].

    Returns (base_margin_m, distance_m) -- distance_m (predicted board
    range from the LiDAR origin) is also exposed for diagnostics."""
    board_center = np.mean(predicted_circle_centers_lidar, axis=0)
    distance_m = float(np.linalg.norm(board_center))

    rotation_error_m = distance_m * np.tan(np.radians(config.rotation_uncertainty_deg))
    raw_margin = config.translation_uncertainty_m + rotation_error_m + config.safety_margin_m
    base_margin = float(np.clip(raw_margin, config.min_margin_m, config.max_margin_m))

    return base_margin, distance_m


def build_margin_schedule(base_margin_m: float, config: GuidedROIConfig) -> list[float]:
    """Ascending, de-duplicated list of ROI margins to try, starting at
    base_margin_m and expanding by config.expansion_factors, with
    config.max_margin_m appended as a final fallback candidate. Every
    value is clipped to <= max_margin_m."""
    max_margin = config.max_margin_m
    candidates = [base_margin_m * factor for factor in config.expansion_factors]
    candidates.append(max_margin)

    clipped = [min(c, max_margin) for c in candidates if c > 0]

    seen: set = set()
    schedule: list[float] = []
    for margin in sorted(clipped):
        key = round(margin, 6)
        if key in seen:
            continue
        seen.add(key)
        schedule.append(margin)
    return schedule


def build_roi_from_predicted_centers(centers_lidar: np.ndarray, margin_m: float) -> ROIConfig:
    """Axis-aligned bounding box around `centers_lidar`, expanded by
    `margin_m` on every side."""
    centers_lidar = np.asarray(centers_lidar, dtype=np.float64)
    mins = centers_lidar.min(axis=0) - margin_m
    maxs = centers_lidar.max(axis=0) + margin_m
    return ROIConfig(
        x_min=float(mins[0]), x_max=float(maxs[0]),
        y_min=float(mins[1]), y_max=float(maxs[1]),
        z_min=float(mins[2]), z_max=float(maxs[2]),
    )
