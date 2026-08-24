"""
evaluation/sanity_gate.py

M0. Projection Sanity Gate (see evaluation_metric_spec.md v0.2-v0.4).

NOT a scored metric. This sits in the Data Quality Assessment stage of the
pipeline, ahead of M2/M3/M4, and answers a different question than they do:
not "how accurate is this calibration" but "is this T/data combination even
in a state where accuracy can be meaningfully measured at all".

Distinct from input/extrinsic.py's verify_extrinsic(), which checks whether
T_CL is a well-formed transform in isolation (valid rotation, finite
translation, plausible units). M0 checks the combination of T_CL + the
actual camera/LiDAR DATA: does projecting real points with this T actually
produce a sane picture, or does it immediately fall apart (nothing lands in
the image, depth is garbage, wildly implausible occlusion pattern)?

Checks implemented:
  1. FOV coverage: fraction of LiDAR points that land inside the image.
  2. Depth distribution sanity: no NaN/Inf, no negative-after-filtering
     depths, and depths fall within a plausible range for the given
     min/max_range_m sensor spec.
  3. Occlusion violation (approximate): using a coarse per-pixel-bucket
     depth buffer, flag points that land far behind another point already
     claiming the same image region -- a rough proxy for "this scene
     projects in a way consistent with basic occlusion," without needing a
     full mesh/z-buffer renderer.

Pass/Fail thresholds are intentionally coarse (this is a gate, not a score):
default thresholds are documented inline and are tunable per call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from geometry.projection import project_lidar_to_image, ProjectionResult
from input.camera import CameraModel
from input.lidar import LidarSensorSpec


DEFAULT_MIN_FOV_COVERAGE = 0.30          # >=30% of points must land in-image
DEFAULT_MAX_OCCLUSION_VIOLATION_RATIO = 0.20   # <=20% of points may be occlusion-violating
DEFAULT_MIN_VALID_POINTS = 500
DEFAULT_OCCLUSION_BUCKET_PX = 8.0        # coarse pixel-bucket size for the occlusion check
DEFAULT_OCCLUSION_DEPTH_MARGIN_M = 0.5   # a point must be this much farther than the
                                          # bucket's nearest point to count as "violating"


@dataclass
class SanityCheckItem:
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None


@dataclass
class SanityGateResult:
    passed: bool
    items: list[SanityCheckItem] = field(default_factory=list)
    num_input_points: int = 0
    num_valid_points: int = 0
    fov_coverage_ratio: float = float("nan")
    occlusion_violation_ratio: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def failed_items(self) -> list[SanityCheckItem]:
        return [i for i in self.items if not i.passed]

    def to_dict(self) -> dict:
        """Plain-dict form for report/builder.py (mirrors the pattern used
        by input/extrinsic.py's SanityReport, kept independent since M0 and
        the extrinsic sanity check serve different pipeline stages)."""
        return {
            "passed": self.passed,
            "num_input_points": self.num_input_points,
            "num_valid_points": self.num_valid_points,
            "fov_coverage_ratio": _safe_float(self.fov_coverage_ratio),
            "occlusion_violation_ratio": _safe_float(self.occlusion_violation_ratio),
            "checks": [
                {"name": i.name, "passed": i.passed, "detail": i.detail, "value": _safe_float(i.value)}
                for i in self.items
            ],
            "warnings": list(self.warnings),
        }


def _safe_float(x):
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _check_depth_distribution(
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    lidar_spec: LidarSensorSpec,
) -> SanityCheckItem:
    """Check for NaN/Inf in the raw point cloud and that in-range points
    exist at all, using the sensor's own min/max_range_m as the plausible
    bound (checked against RAW lidar-frame range, not camera-frame depth,
    since range is a property of the LiDAR measurement itself)."""
    if points_lidar.shape[0] == 0:
        return SanityCheckItem("depth_distribution_valid", False, "Point cloud is empty.", value=0.0)

    finite_mask = np.isfinite(points_lidar).all(axis=1)
    n_nonfinite = int((~finite_mask).sum())
    ranges = np.linalg.norm(points_lidar[finite_mask], axis=1) if finite_mask.any() else np.array([])

    in_range_mask = (ranges >= lidar_spec.min_range_m) & (ranges <= lidar_spec.max_range_m)
    n_in_range = int(in_range_mask.sum())
    frac_in_range = n_in_range / points_lidar.shape[0]

    passed = (n_nonfinite == 0) and frac_in_range > 0.5
    detail = (
        f"{n_nonfinite} non-finite point(s); {n_in_range}/{points_lidar.shape[0]} "
        f"({frac_in_range:.1%}) within sensor range [{lidar_spec.min_range_m}, {lidar_spec.max_range_m}] m."
    )
    return SanityCheckItem("depth_distribution_valid", passed, detail, value=frac_in_range)


def _check_occlusion_violations(
    pixels: np.ndarray,
    depths: np.ndarray,
    image_width: int,
    image_height: int,
    bucket_px: float,
    depth_margin_m: float,
) -> tuple[float, SanityCheckItem]:
    """
    Coarse occlusion sanity check: bucket the image into bucket_px x bucket_px
    cells, find the minimum depth in each cell (the "nearest surface" a real
    camera would actually see there), and flag any point that's more than
    depth_margin_m farther than that cell's minimum as occlusion-violating
    -- i.e. a point that projects behind something already known to be
    closer at essentially the same pixel location, which real depth-first
    visibility would never show. This is a coarse proxy, not a renderer: it
    catches gross misprojection (e.g. a badly wrong T folding far-away
    points on top of near ones), not subtle errors.
    """
    if pixels.shape[0] == 0:
        return float("nan"), SanityCheckItem("occlusion_plausible", False, "No valid points to check.")

    bucket_x = (pixels[:, 0] // bucket_px).astype(int)
    bucket_y = (pixels[:, 1] // bucket_px).astype(int)
    bucket_id = bucket_y * (int(np.ceil(image_width / bucket_px)) + 1) + bucket_x

    # minimum depth per bucket via a fast groupby using argsort
    order = np.argsort(bucket_id)
    sorted_buckets = bucket_id[order]
    sorted_depths = depths[order]
    unique_buckets, first_idx = np.unique(sorted_buckets, return_index=True)
    min_depth_per_bucket = np.minimum.reduceat(sorted_depths, first_idx)
    bucket_to_min = dict(zip(unique_buckets.tolist(), min_depth_per_bucket.tolist()))

    nearest_depth_at_point = np.array([bucket_to_min[b] for b in bucket_id])
    violation_mask = depths > (nearest_depth_at_point + depth_margin_m)
    violation_ratio = float(violation_mask.sum()) / pixels.shape[0]

    passed = violation_ratio <= DEFAULT_MAX_OCCLUSION_VIOLATION_RATIO
    detail = (
        f"{int(violation_mask.sum())}/{pixels.shape[0]} points ({violation_ratio:.1%}) project "
        f">= {depth_margin_m}m behind the nearest point in their {bucket_px:.0f}px image region."
    )
    return violation_ratio, SanityCheckItem("occlusion_plausible", passed, detail, value=violation_ratio)


def run_sanity_gate(
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpec,
    min_fov_coverage: float = DEFAULT_MIN_FOV_COVERAGE,
    min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
    occlusion_bucket_px: float = DEFAULT_OCCLUSION_BUCKET_PX,
    occlusion_depth_margin_m: float = DEFAULT_OCCLUSION_DEPTH_MARGIN_M,
) -> SanityGateResult:
    """
    Run the M0 Projection Sanity Gate for a single frame's point cloud
    against a fixed T_CL and camera model. Returns a SanityGateResult with
    an overall pass/fail plus itemized checks -- callers (e.g. the
    evaluation pipeline driver) should skip or heavily flag M2/M3/M4 results
    for a frame/dataset that fails this gate, since those metrics assume
    the projection is at least structurally sane.
    """
    warnings: list[str] = []
    items: list[SanityCheckItem] = []

    n_input = points_lidar.shape[0]

    depth_check = _check_depth_distribution(points_lidar, T_CL, lidar_spec)
    items.append(depth_check)

    projection: ProjectionResult = project_lidar_to_image(
        points_lidar=points_lidar, T_CL=T_CL, K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height,
        camera_model=camera.projection_model_name(),
    )

    fov_coverage = projection.num_valid_points / n_input if n_input > 0 else 0.0
    fov_passed = fov_coverage >= min_fov_coverage
    items.append(SanityCheckItem(
        "fov_coverage_sufficient", fov_passed,
        f"{projection.num_valid_points}/{n_input} points ({fov_coverage:.1%}) landed inside the "
        f"{camera.width}x{camera.height} image (need >= {min_fov_coverage:.0%}).",
        value=fov_coverage,
    ))

    valid_points_check_passed = projection.num_valid_points >= min_valid_points
    items.append(SanityCheckItem(
        "sufficient_valid_points", valid_points_check_passed,
        f"{projection.num_valid_points} valid projected points (need >= {min_valid_points}).",
        value=float(projection.num_valid_points),
    ))

    if projection.num_valid_points > 0:
        occlusion_ratio, occlusion_item = _check_occlusion_violations(
            projection.pixels, projection.depths, camera.width, camera.height,
            occlusion_bucket_px, occlusion_depth_margin_m,
        )
        items.append(occlusion_item)
    else:
        occlusion_ratio = float("nan")
        items.append(SanityCheckItem(
            "occlusion_plausible", False,
            "Skipped: no valid projected points to check.",
        ))
        warnings.append("No points projected into the image; occlusion check skipped.")

    if not fov_passed:
        warnings.append(
            f"FOV coverage ({fov_coverage:.1%}) is below the minimum ({min_fov_coverage:.0%}). "
            f"Check T_CL direction/units, or that camera and LiDAR actually overlap in this scene."
        )
    if not valid_points_check_passed:
        warnings.append(
            f"Only {projection.num_valid_points} valid points; M2/M3/M4 results from this frame "
            f"would be based on very sparse data."
        )

    overall_passed = all(item.passed for item in items)

    return SanityGateResult(
        passed=overall_passed,
        items=items,
        num_input_points=n_input,
        num_valid_points=projection.num_valid_points,
        fov_coverage_ratio=fov_coverage,
        occlusion_violation_ratio=occlusion_ratio,
        warnings=warnings,
    )
