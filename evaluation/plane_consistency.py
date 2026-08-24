"""
evaluation/plane_consistency.py

Advanced/Phase-5 metric: Plane Consistency (see evaluation_metric_spec.md
section 15). Not part of the MVP scored set.

Idea: fit the dominant plane in the LiDAR point cloud (typically the
ground, or a large wall) via RANSAC, project its inlier points into the
image, and check whether the OUTLINE of that projected region (its convex
hull boundary) lines up with actual image edges.

Note this deliberately does NOT reuse M2's depth-discontinuity edge-point
extraction: a single fitted plane has no internal depth discontinuity by
definition (that's what makes it a plane), so M2's edge-point selection
would always find zero points if applied to a plane's inliers directly.
What IS meaningful for a single flat surface is its silhouette -- where the
surface visually ends in the image (against another object, the horizon,
etc) -- so this metric extracts boundary points via each inlier's distance
to the projected point set's 2D convex hull, then reuses the same
edge-map + distance-transform sampling M2 uses for the actual alignment
check. This keeps the two metrics consistent in HOW they measure alignment
(distance-transform sampling) while using a definition of "edge point"
appropriate to each metric's geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from evaluation.edge_alignment import extract_image_edges, compute_distance_transform, sample_bilinear
from geometry.projection import project_lidar_to_image, ProjectionResult
from input.camera import CameraModel
from quality.noise_floor import (
    LidarSensorSpecForFloor, resolve_floor_inputs, compute_floor, classify,
    M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER,
)


DEFAULT_PLANE_DISTANCE_THRESHOLD_M = 0.05
DEFAULT_RANSAC_ITERATIONS = 300
DEFAULT_MIN_INLIER_RATIO = 0.10
DEFAULT_BOUNDARY_MARGIN_PX = 4.0
DEFAULT_MIN_BOUNDARY_POINTS = 30


@dataclass
class PlaneModel:
    normal: np.ndarray     # (3,) unit normal
    offset: float           # d, such that normal . p + d = 0 for points on the plane
    inlier_mask: np.ndarray  # (N,) bool
    num_inliers: int
    inlier_ratio: float


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold_m: float = DEFAULT_PLANE_DISTANCE_THRESHOLD_M,
    iterations: int = DEFAULT_RANSAC_ITERATIONS,
    seed: int = 0,
) -> Optional[PlaneModel]:
    """
    Fit the dominant plane in `points` (N, 3) via basic 3-point RANSAC.
    Returns None if there are fewer than 3 points or every sampled triple
    was degenerate (collinear).

    This is deliberately minimal (no external RANSAC library): sample 3
    points, form a plane, count inliers within distance_threshold_m, keep
    the best. Adequate for finding one dominant flat surface (ground,
    wall) in a scene; not a general multi-plane segmenter.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return None

    rng = np.random.RandomState(seed)
    best_inlier_mask: Optional[np.ndarray] = None
    best_count = -1
    best_normal = None
    best_offset = None

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue  # degenerate (collinear) sample, skip
        normal = normal / norm_len
        offset = -float(normal @ p0)

        distances = np.abs(points @ normal + offset)
        inlier_mask = distances <= distance_threshold_m
        count = int(inlier_mask.sum())

        if count > best_count:
            best_count = count
            best_inlier_mask = inlier_mask
            best_normal = normal
            best_offset = offset

    if best_inlier_mask is None:
        return None

    return PlaneModel(
        normal=best_normal, offset=best_offset, inlier_mask=best_inlier_mask,
        num_inliers=best_count, inlier_ratio=best_count / n,
    )


def extract_plane_boundary_mask(pixels: np.ndarray, margin_px: float = DEFAULT_BOUNDARY_MARGIN_PX) -> np.ndarray:
    """
    Given (N, 2) projected pixel coordinates for a plane's inlier points,
    return a boolean mask of the points near the 2D convex hull boundary
    of that projected set -- i.e. the plane's visible silhouette in the
    image, which is what should line up with a real image edge (the
    surface's actual boundary against whatever is next to it).
    """
    n = pixels.shape[0]
    if n < 3:
        return np.zeros(n, dtype=bool)

    hull = cv2.convexHull(pixels.astype(np.float32))
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        dist = cv2.pointPolygonTest(hull, (float(pixels[i, 0]), float(pixels[i, 1])), True)
        mask[i] = abs(dist) <= margin_px
    return mask


@dataclass
class PlaneConsistencyResult:
    classification: str    # GOOD | WARNING | BAD | FAIL
    plane_found: bool
    inlier_ratio: float
    num_inliers: int
    num_boundary_points: int = 0
    mean_px: float = float("nan")
    median_px: float = float("nan")
    p95_px: float = float("nan")
    floor_px: float = float("nan")
    plane_normal: Optional[list] = None
    warnings: list[str] = field(default_factory=list)


def evaluate_plane_consistency(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    plane_distance_threshold_m: float = DEFAULT_PLANE_DISTANCE_THRESHOLD_M,
    ransac_iterations: int = DEFAULT_RANSAC_ITERATIONS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    boundary_margin_px: float = DEFAULT_BOUNDARY_MARGIN_PX,
    min_boundary_points: int = DEFAULT_MIN_BOUNDARY_POINTS,
    canny_low: int = 50,
    canny_high: int = 150,
) -> PlaneConsistencyResult:
    """
    Fit the dominant plane in points_lidar, project its inliers into the
    image, extract the projected region's convex-hull boundary points, and
    measure their distance-transform alignment against real image edges
    (same sampling approach as M2, applied to a different point selection).

    FAILs (plane_found=False) if no plane with at least min_inlier_ratio of
    the points can be found. Also FAILs if too few boundary points project
    into the image to form a meaningful statistic.
    """
    warnings: list[str] = []

    plane = fit_plane_ransac(points_lidar, plane_distance_threshold_m, ransac_iterations)

    if plane is None or plane.inlier_ratio < min_inlier_ratio:
        ratio = plane.inlier_ratio if plane else 0.0
        warnings.append(
            f"No dominant plane found with inlier ratio >= {min_inlier_ratio:.0%} "
            f"(best: {ratio:.1%}). Scene may lack a large flat surface (ground/wall) in view."
        )
        return PlaneConsistencyResult(
            classification="FAIL", plane_found=False, inlier_ratio=ratio,
            num_inliers=plane.num_inliers if plane else 0, warnings=warnings,
        )

    inlier_points = points_lidar[plane.inlier_mask]

    projection: ProjectionResult = project_lidar_to_image(
        points_lidar=inlier_points, T_CL=T_CL, K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height,
        camera_model=camera.projection_model_name(),
    )
    if projection.num_valid_points < 3:
        warnings.append(f"Only {projection.num_valid_points} plane inlier(s) projected into the image.")
        return PlaneConsistencyResult(
            classification="FAIL", plane_found=True, inlier_ratio=plane.inlier_ratio,
            num_inliers=plane.num_inliers, plane_normal=plane.normal.tolist(), warnings=warnings,
        )

    boundary_mask = extract_plane_boundary_mask(projection.pixels, boundary_margin_px)
    num_boundary = int(boundary_mask.sum())

    if num_boundary < min_boundary_points:
        warnings.append(
            f"Only {num_boundary} plane-boundary points found (need >= {min_boundary_points}); "
            f"projected plane region may be too small or too thin in this frame."
        )
        return PlaneConsistencyResult(
            classification="FAIL", plane_found=True, inlier_ratio=plane.inlier_ratio,
            num_inliers=plane.num_inliers, num_boundary_points=num_boundary,
            plane_normal=plane.normal.tolist(), warnings=warnings,
        )

    boundary_pixels = projection.pixels[boundary_mask]
    boundary_depths = projection.depths[boundary_mask]

    edge_map = extract_image_edges(image, canny_low, canny_high)
    if edge_map.sum() == 0:
        warnings.append("Canny edge detection found no edges in the image (low-texture scene?).")
        return PlaneConsistencyResult(
            classification="FAIL", plane_found=True, inlier_ratio=plane.inlier_ratio,
            num_inliers=plane.num_inliers, num_boundary_points=num_boundary,
            plane_normal=plane.normal.tolist(), warnings=warnings,
        )

    dt = compute_distance_transform(edge_map)
    errors_px = sample_bilinear(dt, boundary_pixels)

    representative_depth_m = float(np.median(boundary_depths))
    floor_inputs = resolve_floor_inputs(
        fx_px=camera.intrinsics.fx, T_CL=T_CL, lidar_spec=lidar_spec,
        edge_localization_floor_px=camera.edge_localization_floor_px,
    )
    warnings.extend(floor_inputs.fallback_warnings)
    floor_px = compute_floor(floor_inputs, representative_depth_m)

    mean_px = float(np.mean(errors_px))
    classification = classify(mean_px, floor_px, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)

    return PlaneConsistencyResult(
        classification=classification,
        plane_found=True,
        inlier_ratio=plane.inlier_ratio,
        num_inliers=plane.num_inliers,
        num_boundary_points=num_boundary,
        mean_px=mean_px,
        median_px=float(np.median(errors_px)),
        p95_px=float(np.percentile(errors_px, 95)),
        floor_px=floor_px,
        plane_normal=plane.normal.tolist(),
        warnings=warnings,
    )
