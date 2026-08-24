"""
evaluation/edge_alignment.py

M2. Edge Alignment -- the primary MVP scored metric (see
evaluation_metric_spec.md v0.4). Measures how well LiDAR depth-discontinuity
("edge") points, projected into the image via the existing T_CL, align with
actual image edges.

Pipeline:
  1. Project all LiDAR points into the image (geometry.projection).
  2. Identify which projected points sit on a LiDAR-side depth discontinuity
     (i.e. correspond to an object silhouette / occlusion boundary).
  3. Extract image edges (Canny) and build a distance-transform map.
  4. Sample the distance-transform value at each LiDAR edge point's pixel
     location -- this is the per-point alignment error, in pixels.
  5. Aggregate (mean/median/P95) and classify against the sensor-relative
     floor(Z) thresholds (quality.noise_floor), using the GOOD/WARNING/BAD
     multipliers specified for M2 (2x / 5x).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from scipy.spatial import cKDTree

from geometry.projection import project_lidar_to_image, ProjectionResult
from input.camera import CameraModel
from quality.noise_floor import (
    LidarSensorSpecForFloor,
    resolve_floor_inputs,
    compute_floor,
    classify,
    M2_GOOD_MULTIPLIER,
    M2_WARNING_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Step 2: LiDAR-side depth-discontinuity ("edge point") extraction
# ---------------------------------------------------------------------------

def extract_lidar_edge_points(
    pixels: np.ndarray,
    depths: np.ndarray,
    radius_px: float = 3.0,
    depth_jump_threshold_m: float = 0.3,
    min_neighbors: int = 3,
) -> np.ndarray:
    """
    Identify which projected LiDAR points sit on a depth discontinuity,
    without requiring ring/azimuth structure (works on the projected 2D
    pixel positions directly).

    For each point, look at other projected points within `radius_px` in
    image space. If the depth range among those neighbors exceeds
    `depth_jump_threshold_m`, this point sits near an occlusion boundary.
    Only the NEAR-SIDE (foreground, smaller-depth) point in that
    neighborhood is kept as the edge point -- this corresponds to the
    silhouette of the closer object, which is what actually produces a
    visible edge in the image (the far-side surface is occluded, not
    edge-defining).

    Returns a boolean mask over the input pixels/depths arrays.

    Performance note: this vectorizes the per-point neighbor reduction using
    cKDTree.query_pairs(..., output_type="ndarray") to get every within-radius
    point pair as one array computed entirely in C, then np.maximum.at /
    np.minimum.at to reduce per-point neighbor depths -- instead of looping
    over every point in Python (via query_ball_point's ragged per-point
    lists) and calling .max()/.min() on each point's neighbor-depth array
    individually. That per-point Python loop -- and even a from-python-list
    flattening step -- used to dominate this function's runtime (and the
    test suite's) for tens of thousands of points, because both a numpy
    reduction call and a Python-level list/generator iteration carry fixed
    overhead that's large relative to the tiny amount of actual work per
    point.
    """
    n = pixels.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    tree = cKDTree(pixels)
    # Each point is trivially its own neighbor (distance 0), matching
    # query_ball_point's semantics (which always includes the query point
    # itself). query_pairs only returns the *other* pairs (i < j, i != j),
    # so self-pairs are added explicitly below.
    pairs = tree.query_pairs(r=radius_px, output_type="ndarray")

    self_ids = np.arange(n, dtype=np.intp)
    if pairs.shape[0] == 0:
        point_ids = self_ids
        neighbor_ids = self_ids
    else:
        i, j = pairs[:, 0], pairs[:, 1]
        # Each unordered pair (i, j) contributes to both i's and j's
        # neighbor set, plus every point is its own neighbor.
        point_ids = np.concatenate([i, j, self_ids])
        neighbor_ids = np.concatenate([j, i, self_ids])

    neighbor_counts = np.bincount(point_ids, minlength=n)
    neighbor_depths = depths[neighbor_ids]

    max_depth = np.full(n, -np.inf, dtype=np.float64)
    min_depth = np.full(n, np.inf, dtype=np.float64)
    np.maximum.at(max_depth, point_ids, neighbor_depths)
    np.minimum.at(min_depth, point_ids, neighbor_depths)

    depth_range = max_depth - min_depth
    edge_mask = (
        (neighbor_counts >= min_neighbors)
        & (depth_range > depth_jump_threshold_m)
        & (depths <= min_depth + 1e-9)
    )
    return edge_mask


# ---------------------------------------------------------------------------
# Step 3-4: image edge extraction + distance transform + sampling
# ---------------------------------------------------------------------------

def extract_image_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Canny edge detection. Accepts BGR or grayscale; returns a binary
    (0/255) uint8 edge map."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, low_threshold, high_threshold)


def compute_distance_transform(edge_map: np.ndarray) -> np.ndarray:
    """
    Given a binary edge map (255 = edge, 0 = background), compute a
    per-pixel distance-to-nearest-edge map using cv2.distanceTransform.

    cv2.distanceTransform measures distance to the nearest ZERO pixel, so
    we invert the edge map first (edges -> 0, background -> 255).
    """
    inverted = cv2.bitwise_not(edge_map)
    return cv2.distanceTransform(inverted, cv2.DIST_L2, 5)


def sample_bilinear(dt: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D float array `dt` at floating-point (u, v)
    pixel coordinates, clamping to image bounds."""
    h, w = dt.shape
    u = np.clip(pixels[:, 0], 0, w - 1 - 1e-6)
    v = np.clip(pixels[:, 1], 0, h - 1 - 1e-6)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)

    du = u - u0
    dv = v - v0

    top = dt[v0, u0] * (1 - du) + dt[v0, u1] * du
    bottom = dt[v1, u0] * (1 - du) + dt[v1, u1] * du
    return top * (1 - dv) + bottom * dv


# ---------------------------------------------------------------------------
# Result type + top-level entry point
# ---------------------------------------------------------------------------

@dataclass
class EdgeAlignmentResult:
    mean_px: float
    median_px: float
    p95_px: float
    max_px: float
    num_edge_points: int
    num_projected_points: int
    representative_depth_m: float
    floor_px: float
    classification: str  # "GOOD" | "WARNING" | "BAD" | "FAIL"
    warnings: list[str] = field(default_factory=list)

    # per-point data retained for downstream visualization (overlay/heatmap)
    edge_point_pixels: Optional[np.ndarray] = None
    edge_point_errors_px: Optional[np.ndarray] = None


def evaluate_edge_alignment(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    canny_low: int = 50,
    canny_high: int = 150,
    edge_radius_px: float = 3.0,
    depth_jump_threshold_m: float = 0.3,
    min_neighbors: int = 3,
    min_edge_points: int = 100,
) -> EdgeAlignmentResult:
    """
    Compute the M2 Edge Alignment metric for a single synced frame.

    Returns an EdgeAlignmentResult. If there aren't enough valid LiDAR edge
    points (per spec's M2 failure condition), classification is "FAIL" and
    the px statistics are NaN.
    """
    warnings: list[str] = []

    projection: ProjectionResult = project_lidar_to_image(
        points_lidar=points_lidar,
        T_CL=T_CL,
        K=camera.K(),
        dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width,
        image_height=camera.height,
        camera_model=camera.projection_model_name(),
    )

    if projection.num_valid_points == 0:
        return _fail_result(warnings + ["No LiDAR points projected into the image bounds."])

    edge_mask = extract_lidar_edge_points(
        projection.pixels, projection.depths,
        radius_px=edge_radius_px,
        depth_jump_threshold_m=depth_jump_threshold_m,
        min_neighbors=min_neighbors,
    )
    num_edge_points = int(edge_mask.sum())

    if num_edge_points < min_edge_points:
        warnings.append(
            f"Only {num_edge_points} LiDAR edge points found (need >= {min_edge_points}). "
            f"Scene may lack depth discontinuities, or depth_jump_threshold_m may need tuning."
        )
        return _fail_result(warnings, num_projected_points=projection.num_valid_points)

    edge_pixels = projection.pixels[edge_mask]
    edge_depths = projection.depths[edge_mask]

    edge_map = extract_image_edges(image, canny_low, canny_high)
    if edge_map.sum() == 0:
        warnings.append("Canny edge detection found no edges in the image (low-texture scene?).")
        return _fail_result(warnings, num_projected_points=projection.num_valid_points,
                             num_edge_points=num_edge_points)

    dt = compute_distance_transform(edge_map)
    errors_px = sample_bilinear(dt, edge_pixels)

    representative_depth_m = float(np.median(edge_depths))

    floor_inputs = resolve_floor_inputs(
        fx_px=camera.intrinsics.fx,
        T_CL=T_CL,
        lidar_spec=lidar_spec,
        edge_localization_floor_px=camera.edge_localization_floor_px,
    )
    warnings.extend(floor_inputs.fallback_warnings)
    floor_px = compute_floor(floor_inputs, representative_depth_m)

    mean_px = float(np.mean(errors_px))
    median_px = float(np.median(errors_px))
    p95_px = float(np.percentile(errors_px, 95))
    max_px = float(np.max(errors_px))

    classification = classify(mean_px, floor_px, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)

    return EdgeAlignmentResult(
        mean_px=mean_px,
        median_px=median_px,
        p95_px=p95_px,
        max_px=max_px,
        num_edge_points=num_edge_points,
        num_projected_points=projection.num_valid_points,
        representative_depth_m=representative_depth_m,
        floor_px=floor_px,
        classification=classification,
        warnings=warnings,
        edge_point_pixels=edge_pixels,
        edge_point_errors_px=errors_px,
    )


def _fail_result(warnings: list[str], num_projected_points: int = 0, num_edge_points: int = 0) -> EdgeAlignmentResult:
    return EdgeAlignmentResult(
        mean_px=float("nan"),
        median_px=float("nan"),
        p95_px=float("nan"),
        max_px=float("nan"),
        num_edge_points=num_edge_points,
        num_projected_points=num_projected_points,
        representative_depth_m=float("nan"),
        floor_px=float("nan"),
        classification="FAIL",
        warnings=warnings,
    )
