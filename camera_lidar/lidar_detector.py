"""
camera_calibrator.camera_lidar.lidar_detector
=================================================

LiDAR-side FAST-Calib target detection: ROI filter -> RANSAC plane
segmentation -> plane-aligned boundary extraction -> clustering -> circle
fit -> 4 circle-center 3D positions in the LiDAR frame, in a consistent
cyclic (but not yet semantically labeled) order around the rectangle.

Two entry points:
- `detect_lidar_target`: MANUAL ROI -- a caller-supplied box, one plane fit.
- `detect_lidar_target_auto`: AUTO ROI -- no box needed. Searches the whole
  cloud for multiple plane candidates (peeling off floor/wall/vehicle-body
  planes the same way you'd peel layers off an onion) and picks whichever
  candidate's detected circles best match the target's known geometry.

Both share `_detect_circles_on_plane` (boundary -> cluster -> circle-fit ->
geometry-check) so the two ROI modes can never silently drift apart.

Pure numpy/scipy implementation (see camera_lidar/types.py module docstring
for license/provenance notes) -- no PCL/Open3D dependency, matching the
existing point-cloud code in input/lidar.py. Standard building blocks are
hand-written: RANSAC (Fischler & Bolles 1981) for the plane, a k-d-tree
angular-gap boundary test for the edge points, a radius-graph connected-
components pass for clustering, and an algebraic (Kasa) circle fit refined
by nonlinear least squares for each cluster.

LiDAR geometry alone cannot tell "top" from "bottom" of the rectangle (no
marker-ID equivalent), so this module only guarantees a consistent cyclic
traversal order -- camera_lidar.correspondence resolves the remaining
rotation/direction ambiguity against the camera-side centers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from camera_lidar.target_config import TargetConfig
from camera_lidar.types import FailureReason, PointCloudFrame, ROIConfig

_MIN_ROI_POINTS = 30
_PLANE_RANSAC_ITERS = 500
_PLANE_INLIER_THRESHOLD_M = 0.02
_MIN_PLANE_INLIER_RATIO = 0.3
_BOUNDARY_NEIGHBOR_RADIUS_M = 0.05
_BOUNDARY_MIN_NEIGHBORS = 4
_BOUNDARY_ANGULAR_GAP_RAD = np.pi / 2.0
_CLUSTER_EPS_M = 0.05
_MIN_CLUSTER_SIZE = 8
_CIRCLE_FIT_MAX_ERROR_M = 0.02
_CIRCLE_RADIUS_TOLERANCE_RATIO = 0.5
_RECTANGLE_EDGE_TOLERANCE_RATIO = 0.3
# 4 = VALID_FULL, 3 = VALID_PARTIAL (gated downstream by camera_lidar.gates),
# <3 is not enough to even attempt a rigid-transform correspondence.
_MIN_VALID_CIRCLES = 3

# AUTO ROI searches the whole cloud, which is typically much larger than a
# hand-set MANUAL ROI box -- a plane candidate needs fewer inliers relative
# to the whole cloud to still be worth pursuing (a small board is a small
# fraction of a full scene), but still enough to fit a plane/circles reliably.
_AUTO_MAX_PLANES_DEFAULT = 6
_AUTO_MIN_PLANE_INLIERS = 60


@dataclass
class LidarDetectionResult:
    success: bool
    failure_reason: Optional[FailureReason] = None
    circle_centers: Optional[np.ndarray] = None   # (4,3) lidar frame, cyclic order

    roi_point_count: int = 0
    plane_inlier_count: int = 0
    plane_inlier_ratio: float = 0.0
    plane_normal: Optional[np.ndarray] = None
    boundary_point_count: int = 0
    circle_candidate_count: int = 0
    valid_circle_count: int = 0
    circle_fit_errors_m: list[float] = field(default_factory=list)

    # AUTO ROI only: how many plane candidates were tried, and which one
    # (0-indexed, in extraction order) was selected as the target plane.
    plane_candidate_count: int = 0
    selected_plane_index: Optional[int] = None


def _apply_roi(points: np.ndarray, roi: ROIConfig) -> np.ndarray:
    mask = (
        (points[:, 0] >= roi.x_min) & (points[:, 0] <= roi.x_max) &
        (points[:, 1] >= roi.y_min) & (points[:, 1] <= roi.y_max) &
        (points[:, 2] >= roi.z_min) & (points[:, 2] <= roi.z_max)
    )
    return points[mask]


def _fit_plane_ransac(
    points: np.ndarray, rng: np.random.Generator, iterations: int, threshold: float
) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
    """Hand-rolled RANSAC plane fit. Returns (unit normal (3,), d, inlier
    mask) with plane equation normal . p + d = 0, or None if no 3 non-
    degenerate points could seed a plane."""
    n = points.shape[0]
    best_inlier_count = -1
    best_inliers = None
    best_normal = None
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold
        count = int(np.sum(inliers))
        if count > best_inlier_count:
            best_inlier_count, best_inliers, best_normal = count, inliers, normal
    if best_inliers is None:
        return None

    # Refine with a least-squares plane fit (SVD) over the RANSAC inlier set.
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid)
    normal = vh[-1]
    if np.dot(normal, best_normal) < 0:
        normal = -normal
    d = -np.dot(normal, centroid)
    dist = np.abs(points @ normal + d)
    inliers = dist < threshold
    return normal, d, inliers


def _rotation_to_align_z(normal: np.ndarray) -> np.ndarray:
    """3x3 rotation R such that R @ normal ~= [0, 0, 1] (Rodrigues' rotation
    formula for the shortest rotation between two unit vectors)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    s = np.linalg.norm(v)
    c = np.dot(normal, z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def _extract_boundary(points_xy: np.ndarray, radius: float) -> np.ndarray:
    """Boundary points = points whose within-radius neighbors are
    concentrated on one angular side (an interior point has neighbors all
    around it; an edge point has a wide empty angular gap). Returns a
    boolean mask over points_xy."""
    tree = cKDTree(points_xy)
    neighbor_lists = tree.query_ball_point(points_xy, radius)
    is_boundary = np.zeros(len(points_xy), dtype=bool)
    for i, neighbors in enumerate(neighbor_lists):
        neighbors = [j for j in neighbors if j != i]
        if len(neighbors) < _BOUNDARY_MIN_NEIGHBORS:
            is_boundary[i] = True
            continue
        deltas = points_xy[neighbors] - points_xy[i]
        angles = np.sort(np.arctan2(deltas[:, 1], deltas[:, 0]))
        gaps = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
        if gaps.max() > _BOUNDARY_ANGULAR_GAP_RAD:
            is_boundary[i] = True
    return is_boundary


def _cluster_points(points_xy: np.ndarray, eps: float, min_size: int) -> list[np.ndarray]:
    """Radius-graph connected-components clustering (DBSCAN-lite, no
    external dependency beyond scipy's cKDTree).

    Known limitation: if a circle hole sits close enough to the plate's
    own outer edge that both boundaries fall within `eps` of each other
    somewhere, this merges them into one oversized cluster and that circle
    is lost (the merged cluster's radius fit then fails the tolerance
    check in _detect_circles_on_plane, so this fails loudly via
    CIRCLES_NOT_FOUND rather than silently producing a wrong center -- but
    it does mean a board with little margin around its holes, or an
    unusually dense/uniform point cloud, can need a smaller `eps` than the
    default here). A future improvement would classify boundary points by
    curvature sign (convex outer edge vs. concave hole edge) instead of
    relying on spatial separation alone.
    """
    n = points_xy.shape[0]
    tree = cKDTree(points_xy)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component = [i]
        while stack:
            j = stack.pop()
            for k in tree.query_ball_point(points_xy[j], eps):
                if not visited[k]:
                    visited[k] = True
                    stack.append(k)
                    component.append(k)
        if len(component) >= min_size:
            clusters.append(np.array(component))
    return clusters


def _fit_circle(points_xy: np.ndarray) -> Optional[tuple[np.ndarray, float, float]]:
    """Algebraic (Kasa) initial guess, refined by nonlinear least squares.
    Returns (center_xy, radius, rms_fit_error) or None if the algebraic
    solve is degenerate."""
    x, y = points_xy[:, 0], points_xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx0, cy0 = sol[0], sol[1]
    r0_sq = sol[2] + cx0 ** 2 + cy0 ** 2
    if r0_sq <= 0:
        return None
    r0 = np.sqrt(r0_sq)

    def residuals(params):
        cx, cy, r = params
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r

    result = least_squares(residuals, x0=[cx0, cy0, r0])
    cx, cy, r = result.x
    fit_error = float(np.sqrt(np.mean(residuals([cx, cy, r]) ** 2)))
    return np.array([cx, cy]), float(r), fit_error


def _cyclic_order(centers_xy: np.ndarray) -> np.ndarray:
    """Order 4 points into a consistent CCW cyclic order around their
    centroid. Does not attempt to resolve which physical corner (top-left
    etc.) each point is -- camera_lidar.correspondence.match_centers()
    resolves that against the camera-side centers."""
    centroid = centers_xy.mean(axis=0)
    rel = centers_xy - centroid
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    return centers_xy[np.argsort(angles)]


def _triangle_geometry_check(ordered_xy: np.ndarray, target: TargetConfig) -> tuple[bool, float]:
    """3 corners of a rectangle -- any ONE vertex removed -- always form the
    same right triangle: two sides equal to (width, height) in some order,
    and the third side (opposite the removed vertex) equal to the diagonal
    sqrt(width^2 + height^2). This holds regardless of *which* corner is
    missing, so a single check (compare sorted pairwise distances against
    sorted [width, height, diagonal]) covers all 4 possible missing-corner
    cases -- which specific corner is missing gets resolved later, during
    correspondence with the camera side's canonical IDs, not here."""
    edges = sorted(
        float(np.linalg.norm(ordered_xy[j] - ordered_xy[i]))
        for i in range(3) for j in range(i + 1, 3)
    )
    w, h = target.delta_width_circles, target.delta_height_circles
    expected = sorted([w, h]) + [float(np.hypot(w, h))]
    tol = _RECTANGLE_EDGE_TOLERANCE_RATIO

    residual = sum(abs(e - x) for e, x in zip(edges, expected))
    ok = all(abs(e - x) <= tol * x for e, x in zip(edges, expected))
    return ok, residual


def _rectangle_geometry_check(ordered_xy: np.ndarray, target: TargetConfig) -> tuple[bool, float]:
    """A rectangle traversed cyclically has edges alternating [w, h, w, h]
    (or [h, w, h, w] depending on starting corner/direction) -- check the 4
    cyclic edge lengths against the target's known spacing. Returns
    (within_tolerance, total_abs_edge_error) -- the residual lets AUTO ROI
    rank multiple plane candidates that all pass the tolerance check by how
    *closely* they match, not just pass/fail."""
    edges = [float(np.linalg.norm(ordered_xy[(i + 1) % 4] - ordered_xy[i])) for i in range(4)]
    e0, e1, e2, e3 = edges
    w, h = target.delta_width_circles, target.delta_height_circles
    tol = _RECTANGLE_EDGE_TOLERANCE_RATIO

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tol * b

    case1_ok = close(e0, w) and close(e2, w) and close(e1, h) and close(e3, h)
    case1_residual = abs(e0 - w) + abs(e2 - w) + abs(e1 - h) + abs(e3 - h)
    case2_ok = close(e0, h) and close(e2, h) and close(e1, w) and close(e3, w)
    case2_residual = abs(e0 - h) + abs(e2 - h) + abs(e1 - w) + abs(e3 - w)

    if case1_ok and (not case2_ok or case1_residual <= case2_residual):
        return True, case1_residual
    if case2_ok:
        return True, case2_residual
    return False, min(case1_residual, case2_residual)


@dataclass
class _CircleStageResult:
    """Internal result of the boundary->cluster->circle-fit->geometry-check
    pipeline for ONE already-fitted plane. Shared by both detect_lidar_target
    (one caller-supplied plane) and detect_lidar_target_auto (many plane
    candidates from the whole cloud)."""
    success: bool
    failure_reason: Optional[FailureReason] = None
    circle_centers_lidar: Optional[np.ndarray] = None
    boundary_point_count: int = 0
    circle_candidate_count: int = 0
    valid_circle_count: int = 0
    circle_fit_errors_m: list[float] = field(default_factory=list)
    geometry_edge_residual: Optional[float] = None


def _detect_circles_on_plane(plane_points: np.ndarray, normal: np.ndarray, target: TargetConfig) -> _CircleStageResult:
    R_align = _rotation_to_align_z(normal)
    flattened = plane_points @ R_align.T
    z_offset = float(np.mean(flattened[:, 2]))

    boundary_mask = _extract_boundary(flattened[:, :2], _BOUNDARY_NEIGHBOR_RADIUS_M)
    boundary_xy = flattened[boundary_mask][:, :2]
    if boundary_xy.shape[0] < _MIN_VALID_CIRCLES * _MIN_CLUSTER_SIZE:
        return _CircleStageResult(
            success=False, failure_reason=FailureReason.CIRCLES_NOT_FOUND,
            boundary_point_count=int(boundary_xy.shape[0]),
        )

    clusters = _cluster_points(boundary_xy, _CLUSTER_EPS_M, _MIN_CLUSTER_SIZE)
    circle_fits = []
    for component in clusters:
        fit = _fit_circle(boundary_xy[component])
        if fit is None:
            continue
        center_xy, r, err = fit
        if err > _CIRCLE_FIT_MAX_ERROR_M:
            continue
        if abs(r - target.circle_radius) > target.circle_radius * _CIRCLE_RADIUS_TOLERANCE_RATIO:
            continue
        circle_fits.append((center_xy, r, err))

    if len(circle_fits) < _MIN_VALID_CIRCLES:
        return _CircleStageResult(
            success=False, failure_reason=FailureReason.CIRCLES_NOT_FOUND,
            boundary_point_count=int(boundary_xy.shape[0]), circle_candidate_count=len(circle_fits),
        )

    # Keep up to 4 best (lowest fit-error) circle candidates -- 4 if that
    # many passed the filters above, otherwise exactly 3 (a PARTIAL scene;
    # camera_lidar.pipeline decides whether 3 is usable via the Quality/
    # Stability/Duplicate gates, this function only reports what LiDAR
    # geometry itself supports).
    circle_fits.sort(key=lambda c: c[2])
    n_circles = min(4, len(circle_fits))
    best = circle_fits[:n_circles]
    centers_xy = np.array([c[0] for c in best])
    fit_errors = [c[2] for c in best]

    ordered_xy = _cyclic_order(centers_xy)
    if n_circles == 4:
        geometry_ok, geometry_residual = _rectangle_geometry_check(ordered_xy, target)
    else:
        geometry_ok, geometry_residual = _triangle_geometry_check(ordered_xy, target)
    if not geometry_ok:
        return _CircleStageResult(
            success=False, failure_reason=FailureReason.GEOMETRY_MISMATCH,
            boundary_point_count=int(boundary_xy.shape[0]), circle_candidate_count=len(circle_fits),
            valid_circle_count=n_circles, circle_fit_errors_m=fit_errors, geometry_edge_residual=geometry_residual,
        )

    # Back-project the ordered 2D circle centers (on the flattened
    # Z=z_offset plane) into the original LiDAR frame. R_align is
    # orthonormal, so its inverse is its transpose.
    centers_flat = np.column_stack([ordered_xy, np.full(n_circles, z_offset)])
    centers_lidar = centers_flat @ R_align

    return _CircleStageResult(
        success=True, circle_centers_lidar=centers_lidar,
        boundary_point_count=int(boundary_xy.shape[0]), circle_candidate_count=len(circle_fits),
        valid_circle_count=n_circles, circle_fit_errors_m=fit_errors, geometry_edge_residual=geometry_residual,
    )


def detect_lidar_target(
    cloud: PointCloudFrame,
    roi: ROIConfig,
    target: TargetConfig,
    rng_seed: int = 42,
) -> LidarDetectionResult:
    """MANUAL ROI: filter to the caller-supplied box, fit one plane, detect
    circles on it. See detect_lidar_target_auto for the AUTO ROI (no box
    required) alternative."""
    rng = np.random.default_rng(rng_seed)
    points = np.asarray(cloud.points[:, :3], dtype=np.float64)
    roi_points = _apply_roi(points, roi)

    if roi_points.shape[0] < _MIN_ROI_POINTS:
        return LidarDetectionResult(
            success=False, failure_reason=FailureReason.INSUFFICIENT_ROI_POINTS,
            roi_point_count=int(roi_points.shape[0]),
        )

    plane = _fit_plane_ransac(roi_points, rng, _PLANE_RANSAC_ITERS, _PLANE_INLIER_THRESHOLD_M)
    if plane is None:
        return LidarDetectionResult(
            success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND,
            roi_point_count=int(roi_points.shape[0]),
        )
    normal, _d, inlier_mask = plane
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = inlier_count / roi_points.shape[0]
    if inlier_ratio < _MIN_PLANE_INLIER_RATIO or inlier_count < _MIN_ROI_POINTS:
        return LidarDetectionResult(
            success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND,
            roi_point_count=int(roi_points.shape[0]), plane_inlier_count=inlier_count,
            plane_inlier_ratio=inlier_ratio, plane_normal=normal,
        )

    plane_points = roi_points[inlier_mask]
    stage = _detect_circles_on_plane(plane_points, normal, target)
    return LidarDetectionResult(
        success=stage.success,
        failure_reason=stage.failure_reason,
        circle_centers=stage.circle_centers_lidar,
        roi_point_count=int(roi_points.shape[0]),
        plane_inlier_count=inlier_count,
        plane_inlier_ratio=inlier_ratio,
        plane_normal=normal,
        boundary_point_count=stage.boundary_point_count,
        circle_candidate_count=stage.circle_candidate_count,
        valid_circle_count=stage.valid_circle_count,
        circle_fit_errors_m=stage.circle_fit_errors_m,
    )


def _stage_progress(result: LidarDetectionResult) -> tuple:
    """Ranks a *failing* candidate by how far it progressed through the
    pipeline, so if every candidate ultimately fails, the diagnostics shown
    to the user are for the most-informative one (e.g. "3/4 circles found"
    beats "plane too small") rather than an arbitrary one."""
    return (result.valid_circle_count, result.circle_candidate_count,
            result.boundary_point_count, result.plane_inlier_count)


def detect_lidar_target_auto(
    cloud: PointCloudFrame,
    target: TargetConfig,
    max_planes: int = _AUTO_MAX_PLANES_DEFAULT,
    rng_seed: int = 42,
) -> LidarDetectionResult:
    """AUTO ROI: LiDAR-only geometry search, no manual box required.

    Iteratively RANSAC-fits up to `max_planes` planes over the *whole*
    cloud, removing each plane's inliers before searching for the next
    (peeling off floor/wall/vehicle-body planes so the board's own plane
    surfaces as its own candidate). Every candidate is run through the same
    boundary->cluster->circle-fit->geometry-check pipeline MANUAL ROI uses;
    "a RANSAC plane was found" is never treated as "the target was found"
    on its own. Among candidates whose circles do match the target
    geometry, the one with the lowest edge-length residual wins.
    """
    rng = np.random.default_rng(rng_seed)
    points = np.asarray(cloud.points[:, :3], dtype=np.float64)

    if points.shape[0] < _MIN_ROI_POINTS:
        return LidarDetectionResult(
            success=False, failure_reason=FailureReason.INSUFFICIENT_ROI_POINTS,
            roi_point_count=int(points.shape[0]),
        )

    remaining = points
    best_result: Optional[LidarDetectionResult] = None
    best_residual: Optional[float] = None
    candidate_count = 0

    for plane_index in range(max_planes):
        if remaining.shape[0] < _AUTO_MIN_PLANE_INLIERS:
            break
        plane = _fit_plane_ransac(remaining, rng, _PLANE_RANSAC_ITERS, _PLANE_INLIER_THRESHOLD_M)
        if plane is None:
            break
        normal, _d, inlier_mask = plane
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count < _AUTO_MIN_PLANE_INLIERS:
            break
        candidate_count += 1
        plane_points = remaining[inlier_mask]
        inlier_ratio = inlier_count / points.shape[0]

        stage = _detect_circles_on_plane(plane_points, normal, target)
        candidate_result = LidarDetectionResult(
            success=stage.success,
            failure_reason=stage.failure_reason,
            circle_centers=stage.circle_centers_lidar,
            roi_point_count=int(points.shape[0]),
            plane_inlier_count=inlier_count,
            plane_inlier_ratio=inlier_ratio,
            plane_normal=normal,
            boundary_point_count=stage.boundary_point_count,
            circle_candidate_count=stage.circle_candidate_count,
            valid_circle_count=stage.valid_circle_count,
            circle_fit_errors_m=stage.circle_fit_errors_m,
            selected_plane_index=plane_index,
        )

        if stage.success:
            residual = stage.geometry_edge_residual if stage.geometry_edge_residual is not None else 0.0
            if best_result is None or not best_result.success or residual < best_residual:
                best_residual = residual
                best_result = candidate_result
        elif best_result is None or not best_result.success:
            if best_result is None or _stage_progress(candidate_result) > _stage_progress(best_result):
                best_result = candidate_result

        # Peel this plane off and keep searching the rest of the cloud.
        remaining = remaining[~inlier_mask]

    if best_result is None:
        return LidarDetectionResult(
            success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND,
            roi_point_count=int(points.shape[0]), plane_candidate_count=candidate_count,
        )

    best_result.plane_candidate_count = candidate_count
    return best_result
