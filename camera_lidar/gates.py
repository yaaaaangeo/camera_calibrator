"""
camera_calibrator.camera_lidar.gates
========================================

Quality / Stability / Duplicate-Pose / Degenerate-Geometry gates that sit
between "a scene was successfully detected and classified" and "the scene
gets auto-saved". Detecting >=3 common features (camera_lidar.pipeline's
FULL/PARTIAL/INVALID classification) is necessary but never sufficient on
its own -- every one of these gates must also pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from camera_lidar.types import (
    CameraLidarCalibrationResult,
    DegenerateGeometryResult,
    DuplicateGateResult,
    QualityGateResult,
    SceneType,
    StabilityGateResult,
    TargetPose,
)

_MIN_TRIANGLE_HEIGHT_RATIO = 0.15  # triangle height must be >=15% of its longest side to not be "near-collinear"
_MIN_PAIRWISE_DISTANCE_M = 1e-4


@dataclass
class QualityThresholds:
    max_reprojection_error_px: float = 2.0
    min_plane_inlier_ratio: float = 0.5
    max_circle_fit_error_m: float = 0.01
    max_sync_delta_ms: float = 100.0
    reject_ambiguous_partial: bool = True


@dataclass
class StabilityThresholds:
    max_position_change_m: float = 0.02
    max_normal_change_deg: float = 3.0


@dataclass
class DuplicateThresholds:
    min_position_difference_m: float = 0.15
    min_orientation_difference_deg: float = 10.0


def _angle_between(n1: np.ndarray, n2: np.ndarray) -> float:
    cos_a = np.clip(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def evaluate_degenerate_geometry(centers: np.ndarray) -> DegenerateGeometryResult:
    """Only meaningful for a 3-point PARTIAL correspondence: a rigid
    transform is technically solvable from exactly 3 points, but numerically
    unstable if they're close to collinear (rotation about that near-line
    axis is then only weakly constrained by the data)."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape[0] < 3:
        return DegenerateGeometryResult(
            passed=False, triangle_area_m2=0.0, min_pairwise_distance_m=0.0,
            reason="Fewer than 3 points; cannot evaluate triangle geometry.",
        )
    p0, p1, p2 = centers[0], centers[1], centers[2]
    area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
    dists = [float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p2 - p1)), float(np.linalg.norm(p0 - p2))]
    min_dist = min(dists)
    longest = max(dists)
    height_estimate = (2.0 * area / longest) if longest > 1e-9 else 0.0

    passed = height_estimate >= _MIN_TRIANGLE_HEIGHT_RATIO * longest and min_dist > _MIN_PAIRWISE_DISTANCE_M
    reason = None
    if not passed:
        reason = (
            f"Detected 3 features are geometrically degenerate (near-collinear): "
            f"triangle area={area * 1e6:.1f} mm^2, min pairwise distance={min_dist * 1000:.1f} mm."
        )
    return DegenerateGeometryResult(passed=passed, triangle_area_m2=area, min_pairwise_distance_m=min_dist, reason=reason)


def compute_target_pose(centers: np.ndarray) -> TargetPose:
    """Shared pose summary (centroid + best-fit plane normal) for the
    Stability and Duplicate-Pose gates. `centers` should be in a single,
    consistent frame (camera OR lidar -- callers must be consistent)."""
    centers = np.asarray(centers, dtype=np.float64)
    position = centers.mean(axis=0)
    if centers.shape[0] >= 3:
        _, _, vh = np.linalg.svd(centers - position)
        normal = vh[-1]
    else:
        normal = np.array([0.0, 0.0, 1.0])
    return TargetPose(position=position, plane_normal=normal, distance=float(np.linalg.norm(position)))


def evaluate_quality_gate(
    result: CameraLidarCalibrationResult, thresholds: Optional[QualityThresholds] = None,
) -> QualityGateResult:
    thresholds = thresholds or QualityThresholds()
    metrics: dict = {}
    reasons: list[str] = []

    cam = result.camera_detection
    if cam is not None and cam.reprojection_error_px is not None:
        metrics["reprojection_error_px"] = cam.reprojection_error_px
        if cam.reprojection_error_px > thresholds.max_reprojection_error_px:
            reasons.append(
                f"Camera reprojection error too high ({cam.reprojection_error_px:.2f}px "
                f"> {thresholds.max_reprojection_error_px}px)."
            )

    lidar = result.lidar_detection
    if lidar is not None:
        metrics["plane_inlier_ratio"] = lidar.plane_inlier_ratio
        if lidar.plane_inlier_ratio < thresholds.min_plane_inlier_ratio:
            reasons.append(
                f"LiDAR plane inlier ratio too low ({lidar.plane_inlier_ratio * 100:.1f}% "
                f"< {thresholds.min_plane_inlier_ratio * 100:.0f}%)."
            )
        if lidar.circle_fit_errors_m:
            worst = max(lidar.circle_fit_errors_m)
            metrics["worst_circle_fit_error_m"] = worst
            if worst > thresholds.max_circle_fit_error_m:
                reasons.append(
                    f"Worst circle fit RMSE too high ({worst * 1000:.2f}mm "
                    f"> {thresholds.max_circle_fit_error_m * 1000:.0f}mm)."
                )

    # correspondence_ambiguous is set whenever the "target held upright"
    # fallback disambiguation was used instead of a real reference_transform
    # (see correspondence.py) -- that fallback is reliable (empirically
    # verified for the FULL 4/4 case across many rig rotations), so it's
    # only treated as gate-blocking for PARTIAL scenes, where there is
    # inherently thinner geometric evidence (3 points, not 4) and the
    # consequence of a wrong guess is a scene silently poisoning the
    # dataset with a swapped corner.
    if result.correspondence_ambiguous and result.scene_type == SceneType.VALID_PARTIAL and thresholds.reject_ambiguous_partial:
        reasons.append(
            "Partial-scene correspondence rests on the 'target held upright' assumption "
            "(no reference calibration was available to confirm it -- see "
            "camera_lidar/correspondence.py). Capture a FULL (4/4) scene first so later "
            "PARTIAL scenes can be confirmed against it, or move the target so all 4 "
            "features are visible."
        )

    if result.scene_type == SceneType.VALID_PARTIAL:
        degenerate = evaluate_degenerate_geometry(result.lidar_centers) if result.lidar_centers is not None else None
        if degenerate is not None:
            metrics["triangle_area_m2"] = degenerate.triangle_area_m2
            metrics["min_pairwise_distance_m"] = degenerate.min_pairwise_distance_m
            if not degenerate.passed:
                reasons.append(degenerate.reason)

    return QualityGateResult(passed=len(reasons) == 0, reason="; ".join(reasons) if reasons else None, metrics=metrics)


def evaluate_stability_gate(
    candidate_pose: TargetPose, recent_poses: list, thresholds: Optional[StabilityThresholds] = None,
) -> StabilityGateResult:
    thresholds = thresholds or StabilityThresholds()
    if not recent_poses:
        return StabilityGateResult(
            passed=False, reason="No recent observations yet to compare against -- waiting for a stable window.",
        )

    position_changes = [float(np.linalg.norm(candidate_pose.position - p.position)) for p in recent_poses]
    normal_changes = [_angle_between(candidate_pose.plane_normal, p.plane_normal) for p in recent_poses]
    max_position_change = max(position_changes)
    max_normal_change = max(normal_changes)

    passed = (
        max_position_change <= thresholds.max_position_change_m
        and max_normal_change <= thresholds.max_normal_change_deg
    )
    reason = None
    if not passed:
        reason = (
            f"Target is moving: position changed {max_position_change * 1000:.1f}mm "
            f"(limit {thresholds.max_position_change_m * 1000:.0f}mm), plane normal changed "
            f"{max_normal_change:.2f} deg (limit {thresholds.max_normal_change_deg} deg). "
            f"Hold the calibration target still."
        )
    return StabilityGateResult(
        passed=passed, reason=reason, position_change_m=max_position_change, normal_change_deg=max_normal_change,
    )


def evaluate_duplicate_gate(
    candidate_pose: TargetPose, existing_scene_poses: dict, thresholds: Optional[DuplicateThresholds] = None,
) -> DuplicateGateResult:
    """existing_scene_poses: {scene_id: TargetPose} for already-captured
    (included) scenes to compare the candidate against."""
    thresholds = thresholds or DuplicateThresholds()
    if not existing_scene_poses:
        return DuplicateGateResult(passed=True)

    nearest_id: Optional[str] = None
    nearest_pos_diff: Optional[float] = None
    nearest_orient_diff: Optional[float] = None
    for scene_id, pose in existing_scene_poses.items():
        pos_diff = float(np.linalg.norm(candidate_pose.position - pose.position))
        orient_diff = _angle_between(candidate_pose.plane_normal, pose.plane_normal)
        if nearest_pos_diff is None or pos_diff < nearest_pos_diff:
            nearest_id, nearest_pos_diff, nearest_orient_diff = scene_id, pos_diff, orient_diff

    is_duplicate = (
        nearest_pos_diff < thresholds.min_position_difference_m
        and nearest_orient_diff < thresholds.min_orientation_difference_deg
    )
    reason = None
    if is_duplicate:
        reason = (
            f"This target pose is very similar to {nearest_id} "
            f"(position diff {nearest_pos_diff * 1000:.1f}mm, orientation diff {nearest_orient_diff:.2f}deg). "
            f"Move the target to a new position or orientation."
        )
    return DuplicateGateResult(
        passed=not is_duplicate, reason=reason, nearest_scene_id=nearest_id,
        position_difference_m=nearest_pos_diff, orientation_difference_deg=nearest_orient_diff,
    )
