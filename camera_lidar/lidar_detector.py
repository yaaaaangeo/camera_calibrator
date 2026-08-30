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
from itertools import combinations
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from camera_lidar.target_config import TargetConfig
from camera_lidar.types import FailureReason, PointCloudFrame, ROIConfig

_MIN_ROI_POINTS = 30
_MIN_ROI_POINTS_FLOOR = 12
_PLANE_RANSAC_ITERS = 500
_PLANE_INLIER_THRESHOLD_M = 0.02
_MIN_PLANE_INLIER_RATIO = 0.3
_PLANE_RANSAC_MAX_HYPOTHESIS_POINTS = 60_000
# Voxel size RANSAC hypothesis/voting is run at (before the final refine
# step, see `refine_points` in _fit_plane_ransac) -- normalizes point density
# so a merged multi-scan cloud's inlier counts/ratios aren't biased by
# whichever region of a candidate plane happens to have denser overlap.
_RANSAC_VOXEL_SIZE_M = _PLANE_INLIER_THRESHOLD_M
_BOUNDARY_NEIGHBOR_RADIUS_M = 0.05
_BOUNDARY_MIN_NEIGHBORS = 4
_BOUNDARY_ANGULAR_GAP_RAD = np.pi / 2.0
_CLUSTER_EPS_M = 0.05
_MIN_CLUSTER_SIZE = 8
_MIN_CLUSTER_SIZE_FLOOR = 5
_CIRCLE_FIT_MAX_ERROR_M = 0.02
_CIRCLE_RADIUS_TOLERANCE_RATIO = 0.5
_RECTANGLE_EDGE_TOLERANCE_RATIO = 0.3
_BOUNDARY_RADIUS_SCALES = (1.0, 0.75, 0.5, 1.25, 1.5)
_CLUSTER_EPS_SCALES = (1.0, 0.75, 0.5, 1.25)
_MAX_GEOMETRY_CANDIDATES = 12
# 4 = VALID_FULL, 3 = VALID_PARTIAL (gated downstream by camera_lidar.gates),
# <3 is not enough to even attempt a rigid-transform correspondence.
_MIN_VALID_CIRCLES = 3

# Density-adaptive parameter estimation (see _estimate_point_spacing /
# _estimate_ambient_spacing_3d / _adaptive_min_point_count): a single set of
# fixed-metric constants can't fit both a sparse single 64-channel scan and a
# densely merged multi-scan cloud, so boundary/cluster/count thresholds are
# derived from the point cloud's own local spacing at detection time, and
# only clipped into [floor, default-above-constant] so already-working dense
# cases keep their original behavior.
_SPACING_ESTIMATE_K = 5
_MIN_ESTIMATED_SPACING_M = 0.001
_MAX_ESTIMATED_SPACING_M = _BOUNDARY_NEIGHBOR_RADIUS_M * 3.0
_BOUNDARY_RADIUS_SPACING_MULTIPLIER = 2.5
_CLUSTER_EPS_SPACING_MULTIPLIER = 2.0
_MIN_CLUSTER_SIZE_CIRCUMFERENCE_FRACTION = 0.3
_SPACING_ESTIMATE_K_3D = 5
_SPACING_ESTIMATE_MAX_SAMPLE = 5_000
_EXPECTED_POINT_COUNT_COVERAGE = 0.25

# AUTO ROI searches the whole cloud, which is typically much larger than a
# hand-set MANUAL ROI box -- a plane candidate needs fewer inliers relative
# to the whole cloud to still be worth pursuing (a small board is a small
# fraction of a full scene), but still enough to fit a plane/circles reliably.
# A cluttered real scene (room/vehicle interior) can easily have more than 6
# large flat surfaces (floor, ceiling, several walls, vehicle body panels)
# competing with the board for RANSAC's largest-plane-first ordering -- 6 was
# tuned against small synthetic/lab scenes and can exhaust its budget before
# ever reaching the board's own (much smaller) plane. Extent-rejected decoy
# planes (_plane_extent_exceeds_auto_search_window) skip the expensive
# boundary-tracing stage, so searching deeper is cheap.
_AUTO_MAX_PLANES_DEFAULT = 20
_AUTO_MIN_PLANE_INLIERS = 60
_AUTO_MIN_PLANE_INLIERS_FLOOR = 20
_AUTO_PLANE_EXTENT_SCALE = 3.0
_AUTO_MIN_PLANE_EXTENT_LIMIT_M = 2.0


class _Cancelled(Exception):
    pass


def _check_cancel(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and cancel_check():
        raise _Cancelled


@dataclass
class LidarDetectionResult:
    success: bool
    failure_reason: Optional[FailureReason] = None
    circle_centers: Optional[np.ndarray] = None   # (4,3) lidar frame, cyclic order

    roi_point_count: int = 0
    plane_inlier_count: int = 0
    plane_inlier_ratio: float = 0.0
    plane_normal: Optional[np.ndarray] = None
    # Mean of the plane's own inlier points (LiDAR frame) -- lets a caller
    # sanity-check "is this plane actually where the board should be" (near
    # the sensor, roughly in front of the camera) vs. a decoy environmental
    # surface (a wall/floor centroid is typically much farther out) without
    # needing the full inlier point array.
    plane_centroid: Optional[np.ndarray] = None
    boundary_point_count: int = 0
    circle_candidate_count: int = 0
    valid_circle_count: int = 0
    circle_fit_errors_m: list[float] = field(default_factory=list)

    # AUTO ROI only: how many plane candidates were tried, and which one
    # (0-indexed, in extraction order) was selected as the target plane.
    plane_candidate_count: int = 0
    selected_plane_index: Optional[int] = None


@dataclass
class PlaneCandidateInfo:
    """Diagnostic snapshot of ONE plane candidate tried by
    detect_lidar_target_auto's plane-peeling search -- passed to the
    optional `on_plane_candidate` callback for every candidate (not just the
    one eventually selected), so a caller can inspect why AUTO ROI picked
    the plane it did instead of the board (e.g. "candidate 0 is a huge, far
    plane that got extent-rejected before boundary tracing ever ran")."""
    index: int
    centroid: np.ndarray                       # (3,) LiDAR frame
    normal: np.ndarray                         # (3,) unit normal, LiDAR frame
    inlier_count: int
    inlier_ratio: float                        # inlier_count / whole-cloud point count
    extent_xy: tuple[float, float]              # plane-aligned in-plane (width, height)
    extent_rejected: bool                      # True if too large to be the board -- circle detection was skipped entirely
    points: np.ndarray                         # (inlier_count, 3) LiDAR frame, for export/visualization
    stage: Optional[_CircleStageResult] = None  # None iff extent_rejected


def _finite_points(points: np.ndarray) -> np.ndarray:
    """Drops NaN/Inf rows. Real LiDAR drivers commonly pad organized/no-return
    points with NaN (or, on MANUAL ROI, these already get excluded implicitly
    since `_apply_roi`'s >=/<= comparisons are False for NaN) -- but AUTO ROI
    has no ROI box to filter through, so without this, invalid points would
    reach the ambient-density estimate (_adaptive_min_point_count) and the
    RANSAC search directly."""
    return points[np.all(np.isfinite(points), axis=1)]


def _apply_roi(points: np.ndarray, roi: ROIConfig) -> np.ndarray:
    mask = (
        (points[:, 0] >= roi.x_min) & (points[:, 0] <= roi.x_max) &
        (points[:, 1] >= roi.y_min) & (points[:, 1] <= roi.y_max) &
        (points[:, 2] >= roi.z_min) & (points[:, 2] <= roi.z_max)
    )
    return points[mask]


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Grid voxel downsampling (one centroid per occupied voxel). A merged
    multi-scan cloud is dense where scans overlap and sparse elsewhere;
    feeding that directly into RANSAC biases both the hypothesis vote and
    the inlier count toward whichever region of a candidate plane happens to
    be denser, not toward how well-supported the plane itself is. Intended
    to feed only the RANSAC hypothesis/voting step -- pass the caller's
    original full-resolution points as `_fit_plane_ransac`'s `refine_points`
    so the returned inlier mask and downstream circle fit keep full
    precision."""
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def _estimate_point_spacing(points_xy: np.ndarray, k: int = _SPACING_ESTIMATE_K) -> float:
    """Median distance to each point's k-th nearest neighbor in the
    plane-projected 2D set, clipped to a sane range. Used to scale the
    boundary/cluster-eps/min-cluster-size parameters to the plane's actual
    point density instead of a fixed metric constant -- a single estimate
    lets the same code handle a sparse single 64-channel scan and a densely
    merged multi-scan cloud."""
    n = points_xy.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return _BOUNDARY_NEIGHBOR_RADIUS_M
    tree = cKDTree(points_xy)
    dist, _ = tree.query(points_xy, k=k_eff + 1)
    spacing = float(np.median(dist[:, -1]))
    return float(np.clip(spacing, _MIN_ESTIMATED_SPACING_M, _MAX_ESTIMATED_SPACING_M))


def _estimate_ambient_spacing_3d(points: np.ndarray, k: int = _SPACING_ESTIMATE_K_3D) -> float:
    """Median 3D k-NN spacing over a (possibly strided-down) sample of the
    whole cloud -- a rough proxy for local LiDAR point density, used only to
    decide how many points a target-sized plane should realistically
    contain (see _adaptive_min_point_count). A fixed point-count floor
    either rejects genuine far-range/sparse detections or lets tiny noise
    clusters through, depending on which side of the true density it lands
    on for a given scan."""
    n = points.shape[0]
    if n > _SPACING_ESTIMATE_MAX_SAMPLE:
        stride = max(1, n // _SPACING_ESTIMATE_MAX_SAMPLE)
        points = points[::stride]
        n = points.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return _PLANE_INLIER_THRESHOLD_M
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=k_eff + 1)
    return float(np.median(dist[:, -1]))


def _adaptive_min_point_count(points: np.ndarray, target: TargetConfig, floor: int, default: int) -> int:
    """Scales a fixed point-count gate to the cloud's own ambient density:
    the same absolute count that's negligible noise for a dense merged cloud
    can be the entire available signal for a single sparse scan of a small,
    distant board. Clipped to [floor, default] so already-working dense
    cases are unaffected -- this only ever lowers the gate for sparse data,
    never raises it above the original fixed constant."""
    spacing = _estimate_ambient_spacing_3d(points)
    board_area = (
        max(target.delta_width_qr_center, target.delta_width_circles) *
        max(target.delta_height_qr_center, target.delta_height_circles)
    )
    expected = board_area / (max(spacing, 1e-6) ** 2) * _EXPECTED_POINT_COUNT_COVERAGE
    return int(np.clip(expected, floor, default))


def _fit_plane_ransac(
    points: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    threshold: float,
    cancel_check: Optional[Callable[[], bool]] = None,
    refine_points: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
    """Hand-rolled RANSAC plane fit. `points` drives hypothesis generation
    and inlier voting -- pass a voxel-downsampled cloud here to keep dense
    regions of a merged multi-scan cloud from dominating the vote. The
    returned (normal, d, inlier_mask) is always refined and evaluated
    against `refine_points` (defaults to `points`), so the mask indexes
    whichever array the caller actually wants inliers from (typically the
    original, full-resolution ROI/remaining-cloud array, for full-precision
    downstream circle fitting). Returns None if no 3 non-degenerate points
    could seed a plane."""
    n = points.shape[0]
    if n > _PLANE_RANSAC_MAX_HYPOTHESIS_POINTS:
        hypothesis_idx = rng.choice(n, size=_PLANE_RANSAC_MAX_HYPOTHESIS_POINTS, replace=False)
        hypothesis_points = points[hypothesis_idx]
    else:
        hypothesis_points = points
    hypothesis_n = hypothesis_points.shape[0]

    best_inlier_count = -1
    best_normal = None
    best_d = None
    for iteration in range(iterations):
        if iteration % 16 == 0:
            _check_cancel(cancel_check)
        idx = rng.choice(hypothesis_n, size=3, replace=False)
        p0, p1, p2 = hypothesis_points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)
        dist = np.abs(hypothesis_points @ normal + d)
        inliers = dist < threshold
        count = int(np.sum(inliers))
        if count > best_inlier_count:
            best_inlier_count, best_normal, best_d = count, normal, d
    if best_normal is None or best_d is None:
        return None

    # Refine with a least-squares plane fit (SVD) over the FULL inlier set,
    # evaluated against `refine_points` (defaults to `points`). The
    # hypothesis search may use a bounded-size/voxel-downsampled subset for
    # speed and density-unbiased voting, but the returned plane/inlier mask
    # is always computed from refine_points.
    refine_source = points if refine_points is None else refine_points
    dist = np.abs(refine_source @ best_normal + best_d)
    best_inliers = dist < threshold
    inlier_pts = refine_source[best_inliers]
    if inlier_pts.shape[0] < 3:
        return None
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[-1]
    if np.dot(normal, best_normal) < 0:
        normal = -normal
    d = -np.dot(normal, centroid)
    dist = np.abs(refine_source @ normal + d)
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


def _extract_boundary(
    points_xy: np.ndarray,
    radius: float,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Boundary points = points whose within-radius neighbors are
    concentrated on one angular side (an interior point has neighbors all
    around it; an edge point has a wide empty angular gap). Returns a
    boolean mask over points_xy."""
    tree = cKDTree(points_xy)
    is_boundary = np.zeros(len(points_xy), dtype=bool)
    for i, point_xy in enumerate(points_xy):
        if i % 256 == 0:
            _check_cancel(cancel_check)
        # Query one point at a time.  Batched radius queries can materialize
        # an enormous all-neighbor object for dense LiDAR frames before this
        # loop even starts, while the per-point form keeps the exact same
        # boundary decision with bounded peak memory.
        neighbors = tree.query_ball_point(point_xy, radius)
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


def _cluster_points(
    points_xy: np.ndarray,
    eps: float,
    min_size: int,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[np.ndarray]:
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
        if i % 256 == 0:
            _check_cancel(cancel_check)
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


def _best_geometry_consistent_fit(
    circle_fits: list[tuple[np.ndarray, float, float]],
    target: TargetConfig,
) -> Optional[tuple[np.ndarray, list[float], int, float]]:
    circle_fits = sorted(circle_fits, key=lambda c: c[2])[:_MAX_GEOMETRY_CANDIDATES]
    best = None

    for n_circles in (4, 3):
        if len(circle_fits) < n_circles:
            continue
        for combo in combinations(circle_fits, n_circles):
            centers_xy = np.array([c[0] for c in combo])
            ordered_xy = _cyclic_order(centers_xy)
            if n_circles == 4:
                geometry_ok, geometry_residual = _rectangle_geometry_check(ordered_xy, target)
            else:
                geometry_ok, geometry_residual = _triangle_geometry_check(ordered_xy, target)
            if not geometry_ok:
                continue
            fit_errors = [c[2] for c in combo]
            score = (n_circles, -geometry_residual, -sum(fit_errors))
            if best is None or score > best[0]:
                best = (score, ordered_xy, fit_errors, n_circles, geometry_residual)
        if best is not None and best[3] == 4:
            break

    if best is None:
        return None
    _score, ordered_xy, fit_errors, n_circles, geometry_residual = best
    return ordered_xy, fit_errors, n_circles, geometry_residual


def _detect_circles_with_tuning(
    flattened: np.ndarray,
    z_offset: float,
    R_align: np.ndarray,
    target: TargetConfig,
    boundary_radius: float,
    cluster_eps: float,
    min_cluster_size: int,
    cancel_check: Optional[Callable[[], bool]],
) -> _CircleStageResult:
    boundary_mask = _extract_boundary(flattened[:, :2], boundary_radius, cancel_check=cancel_check)
    boundary_xy = flattened[boundary_mask][:, :2]
    if boundary_xy.shape[0] < _MIN_VALID_CIRCLES * min_cluster_size:
        return _CircleStageResult(
            success=False,
            failure_reason=FailureReason.CIRCLES_NOT_FOUND,
            boundary_point_count=int(boundary_xy.shape[0]),
        )

    clusters = _cluster_points(boundary_xy, cluster_eps, min_cluster_size, cancel_check=cancel_check)
    circle_fits = []
    for component in clusters:
        _check_cancel(cancel_check)
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

    selected = _best_geometry_consistent_fit(circle_fits, target)
    if selected is None:
        n_circles = min(4, len(circle_fits))
        fit_errors = [c[2] for c in sorted(circle_fits, key=lambda c: c[2])[:n_circles]]
        return _CircleStageResult(
            success=False, failure_reason=FailureReason.GEOMETRY_MISMATCH,
            boundary_point_count=int(boundary_xy.shape[0]), circle_candidate_count=len(circle_fits),
            valid_circle_count=n_circles, circle_fit_errors_m=fit_errors,
        )
    ordered_xy, fit_errors, n_circles, geometry_residual = selected

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


def _better_circle_stage(candidate: _CircleStageResult, current: Optional[_CircleStageResult]) -> bool:
    if current is None:
        return True
    if candidate.success != current.success:
        return candidate.success
    if candidate.valid_circle_count != current.valid_circle_count:
        return candidate.valid_circle_count > current.valid_circle_count
    if candidate.circle_candidate_count != current.circle_candidate_count:
        return candidate.circle_candidate_count > current.circle_candidate_count
    if candidate.boundary_point_count != current.boundary_point_count:
        return candidate.boundary_point_count > current.boundary_point_count
    if candidate.geometry_edge_residual is None:
        return False
    if current.geometry_edge_residual is None:
        return True
    return candidate.geometry_edge_residual < current.geometry_edge_residual


def _detect_circles_on_plane(
    plane_points: np.ndarray,
    normal: np.ndarray,
    target: TargetConfig,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> _CircleStageResult:
    R_align = _rotation_to_align_z(normal)
    flattened = plane_points @ R_align.T
    z_offset = float(np.mean(flattened[:, 2]))
    best_result: Optional[_CircleStageResult] = None

    # Density-adaptive base parameters (see _estimate_point_spacing):
    # _BOUNDARY_RADIUS_SCALES / _CLUSTER_EPS_SCALES below now sweep AROUND
    # these plane-specific estimates instead of around fixed metric
    # constants, so the same sweep works for a sparse single-scan plane and
    # a densely merged one. Floored at the original fixed constants -- for a
    # cloud at least as dense as what those constants were tuned for, this
    # must reduce to exactly the old behavior; adaptation should only ever
    # WIDEN the radius/eps for genuinely sparse data, never narrow it below
    # the validated default (a narrower radius makes boundary extraction
    # MORE sensitive to incidental density variations that aren't real hole
    # edges, e.g. a solid filled disc sampled at a different density than
    # its surrounding plate).
    spacing = _estimate_point_spacing(flattened[:, :2])
    base_boundary_radius = max(spacing * _BOUNDARY_RADIUS_SPACING_MULTIPLIER, _BOUNDARY_NEIGHBOR_RADIUS_M)
    base_cluster_eps = max(spacing * _CLUSTER_EPS_SPACING_MULTIPLIER, _CLUSTER_EPS_M)
    expected_circumference_points = (2.0 * np.pi * target.circle_radius) / spacing
    min_cluster_size = int(np.clip(
        round(expected_circumference_points * _MIN_CLUSTER_SIZE_CIRCUMFERENCE_FRACTION),
        _MIN_CLUSTER_SIZE_FLOOR,
        _MIN_CLUSTER_SIZE,
    ))

    for boundary_scale in _BOUNDARY_RADIUS_SCALES:
        for eps_scale in _CLUSTER_EPS_SCALES:
            _check_cancel(cancel_check)
            result = _detect_circles_with_tuning(
                flattened,
                z_offset,
                R_align,
                target,
                base_boundary_radius * boundary_scale,
                base_cluster_eps * eps_scale,
                min_cluster_size,
                cancel_check,
            )
            if _better_circle_stage(result, best_result):
                best_result = result
            if result.success and result.valid_circle_count == 4:
                return result

    return best_result or _CircleStageResult(
        success=False,
        failure_reason=FailureReason.CIRCLES_NOT_FOUND,
    )


def _plane_extent_exceeds_auto_search_window(
    plane_points: np.ndarray, normal: np.ndarray, target: TargetConfig, min_plane_inliers: int
) -> bool:
    """Reject huge AUTO-ROI decoy planes before expensive boundary tracing.

    A FAST-Calib target plane should be on the order of the configured board
    geometry.  Floors/walls can have hundreds of thousands of inliers; running
    hole-boundary extraction on those planes is both slow and not useful.
    """
    if plane_points.shape[0] < min_plane_inliers:
        return False

    R_align = _rotation_to_align_z(normal)
    xy = (plane_points @ R_align.T)[:, :2]
    extent = np.ptp(xy, axis=0)
    board_width = max(
        target.delta_width_qr_center + 2.0 * target.marker_size,
        target.delta_width_circles + 2.0 * target.circle_radius,
    )
    board_height = max(
        target.delta_height_qr_center + 2.0 * target.marker_size,
        target.delta_height_circles + 2.0 * target.circle_radius,
    )
    limit_x = max(_AUTO_MIN_PLANE_EXTENT_LIMIT_M, board_width * _AUTO_PLANE_EXTENT_SCALE)
    limit_y = max(_AUTO_MIN_PLANE_EXTENT_LIMIT_M, board_height * _AUTO_PLANE_EXTENT_SCALE)
    return bool(extent[0] > limit_x or extent[1] > limit_y)


def detect_lidar_target(
    cloud: PointCloudFrame,
    roi: ROIConfig,
    target: TargetConfig,
    rng_seed: int = 42,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> LidarDetectionResult:
    """MANUAL ROI: filter to the caller-supplied box, fit one plane, detect
    circles on it. See detect_lidar_target_auto for the AUTO ROI (no box
    required) alternative."""
    try:
        rng = np.random.default_rng(rng_seed)
        points = _finite_points(np.asarray(cloud.points[:, :3], dtype=np.float64))
        roi_points = _apply_roi(points, roi)

        min_roi_points = _adaptive_min_point_count(points, target, _MIN_ROI_POINTS_FLOOR, _MIN_ROI_POINTS)
        if roi_points.shape[0] < min_roi_points:
            return LidarDetectionResult(
                success=False, failure_reason=FailureReason.INSUFFICIENT_ROI_POINTS,
                roi_point_count=int(roi_points.shape[0]),
            )

        ransac_points = _voxel_downsample(roi_points, _RANSAC_VOXEL_SIZE_M)
        plane = _fit_plane_ransac(
            ransac_points, rng, _PLANE_RANSAC_ITERS, _PLANE_INLIER_THRESHOLD_M,
            cancel_check=cancel_check, refine_points=roi_points,
        )
        if plane is None:
            return LidarDetectionResult(
                success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND,
                roi_point_count=int(roi_points.shape[0]),
            )
        normal, _d, inlier_mask = plane
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = inlier_count / roi_points.shape[0]
        plane_points = roi_points[inlier_mask]
        plane_centroid = plane_points.mean(axis=0) if inlier_count > 0 else None
        if inlier_ratio < _MIN_PLANE_INLIER_RATIO or inlier_count < min_roi_points:
            return LidarDetectionResult(
                success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND,
                roi_point_count=int(roi_points.shape[0]), plane_inlier_count=inlier_count,
                plane_inlier_ratio=inlier_ratio, plane_normal=normal, plane_centroid=plane_centroid,
            )

        stage = _detect_circles_on_plane(plane_points, normal, target, cancel_check=cancel_check)
        return LidarDetectionResult(
            success=stage.success,
            failure_reason=stage.failure_reason,
            circle_centers=stage.circle_centers_lidar,
            roi_point_count=int(roi_points.shape[0]),
            plane_inlier_count=inlier_count,
            plane_inlier_ratio=inlier_ratio,
            plane_normal=normal,
            plane_centroid=plane_centroid,
            boundary_point_count=stage.boundary_point_count,
            circle_candidate_count=stage.circle_candidate_count,
            valid_circle_count=stage.valid_circle_count,
            circle_fit_errors_m=stage.circle_fit_errors_m,
        )
    except _Cancelled:
        return LidarDetectionResult(success=False, failure_reason=FailureReason.CANCELLED)


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
    cancel_check: Optional[Callable[[], bool]] = None,
    on_plane_candidate: Optional[Callable[[PlaneCandidateInfo], None]] = None,
    search_roi: Optional[ROIConfig] = None,
) -> LidarDetectionResult:
    """AUTO ROI: LiDAR-only geometry search, no manual box required.

    Iteratively RANSAC-fits up to `max_planes` planes over the *whole*
    cloud (or, if `search_roi` is given, the subset of the cloud inside
    that box -- see camera_lidar.guided_roi/GUIDED AUTO mode), removing
    each plane's inliers before searching for the next (peeling off
    floor/wall/vehicle-body planes so the board's own plane surfaces as its
    own candidate). Every candidate is run through the same
    boundary->cluster->circle-fit->geometry-check pipeline MANUAL ROI uses;
    "a RANSAC plane was found" is never treated as "the target was found"
    on its own. Among candidates whose circles do match the target
    geometry, the one with the lowest edge-length residual wins.

    on_plane_candidate: optional diagnostic hook, called once per candidate
    (in extraction order, including ones later peeled off and discarded)
    with a PlaneCandidateInfo -- lets a caller inspect *why* AUTO ROI picked
    the plane it did (e.g. dump every candidate's centroid/normal/extent, or
    export each candidate's inlier points for external visualization) beyond
    what the returned LidarDetectionResult keeps for only the winner. Has no
    effect on detection itself.

    search_roi: optional ROIConfig box to pre-filter the cloud before the
    multi-plane search -- when None (the default), behavior is identical to
    the original whole-cloud AUTO search.
    """
    try:
        rng = np.random.default_rng(rng_seed)
        all_points = _finite_points(np.asarray(cloud.points[:, :3], dtype=np.float64))
        points = all_points if search_roi is None else _apply_roi(all_points, search_roi)

        min_roi_points = _adaptive_min_point_count(points, target, _MIN_ROI_POINTS_FLOOR, _MIN_ROI_POINTS)
        if points.shape[0] < min_roi_points:
            return LidarDetectionResult(
                success=False, failure_reason=FailureReason.INSUFFICIENT_ROI_POINTS,
                roi_point_count=int(points.shape[0]),
            )

        min_plane_inliers = _adaptive_min_point_count(
            points, target, _AUTO_MIN_PLANE_INLIERS_FLOOR, _AUTO_MIN_PLANE_INLIERS,
        )

        remaining = points
        best_result: Optional[LidarDetectionResult] = None
        best_residual: Optional[float] = None
        candidate_count = 0

        for plane_index in range(max_planes):
            if remaining.shape[0] < min_plane_inliers:
                break
            _check_cancel(cancel_check)
            ransac_points = _voxel_downsample(remaining, _RANSAC_VOXEL_SIZE_M)
            plane = _fit_plane_ransac(
                ransac_points, rng, _PLANE_RANSAC_ITERS, _PLANE_INLIER_THRESHOLD_M,
                cancel_check=cancel_check, refine_points=remaining,
            )
            if plane is None:
                break
            normal, _d, inlier_mask = plane
            inlier_count = int(np.sum(inlier_mask))
            if inlier_count < min_plane_inliers:
                break
            candidate_count += 1
            plane_points = remaining[inlier_mask]
            plane_centroid = plane_points.mean(axis=0)
            inlier_ratio = inlier_count / points.shape[0]

            extent_rejected = _plane_extent_exceeds_auto_search_window(
                plane_points, normal, target, min_plane_inliers,
            )
            stage: Optional[_CircleStageResult] = None
            if not extent_rejected:
                stage = _detect_circles_on_plane(plane_points, normal, target, cancel_check=cancel_check)

            if on_plane_candidate is not None:
                R_align = _rotation_to_align_z(normal)
                extent_xy = tuple(np.ptp((plane_points @ R_align.T)[:, :2], axis=0).tolist())
                on_plane_candidate(PlaneCandidateInfo(
                    index=plane_index,
                    centroid=plane_centroid,
                    normal=normal,
                    inlier_count=inlier_count,
                    inlier_ratio=inlier_ratio,
                    extent_xy=extent_xy,
                    extent_rejected=extent_rejected,
                    points=plane_points,
                    stage=stage,
                ))

            if extent_rejected:
                candidate_result = LidarDetectionResult(
                    success=False,
                    failure_reason=FailureReason.CIRCLES_NOT_FOUND,
                    roi_point_count=int(points.shape[0]),
                    plane_inlier_count=inlier_count,
                    plane_inlier_ratio=inlier_ratio,
                    plane_normal=normal,
                    plane_centroid=plane_centroid,
                    selected_plane_index=plane_index,
                )
                if best_result is None:
                    best_result = candidate_result
                remaining = remaining[~inlier_mask]
                continue

            candidate_result = LidarDetectionResult(
                success=stage.success,
                failure_reason=stage.failure_reason,
                circle_centers=stage.circle_centers_lidar,
                roi_point_count=int(points.shape[0]),
                plane_inlier_count=inlier_count,
                plane_inlier_ratio=inlier_ratio,
                plane_normal=normal,
                plane_centroid=plane_centroid,
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
    except _Cancelled:
        return LidarDetectionResult(success=False, failure_reason=FailureReason.CANCELLED)
