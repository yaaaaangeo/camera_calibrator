"""
camera_calibrator.camera_lidar.multi_scene
==============================================

Multi-Scene FAST-Calib: pools correspondences from multiple captured
Scenes and solves ONE joint rigid transform, then reports how well that
joint transform fits each scene individually (so outlier scenes can be
spotted and excluded) -- per the explicit requirement that a single-scene
result is provisional/debug-only, never the final answer.

The joint solve itself needs no new math: concatenating every scene's
matched (lidar_centers, camera_centers) 3D-3D point pairs and running the
same closed-form Kabsch/SVD solve (camera_lidar.solver.solve_rigid_transform)
used for a single scene is the standard multi-view rigid-registration
approach -- every scene's 4 points contribute equally to one least-squares
solve, rather than averaging N independent per-scene transforms.

Per-scene detection reuses camera_lidar.pipeline.calibrate_single_scene
(not a separate implementation) so a scene's single-scene preview (shown
at capture time) and its contribution to the joint solve are always
computed identically.

Calibration Policy (STRICT / FLEXIBLE) is implemented as a single solver
(this module) fed a different SUBSET of scenes -- never as two separate
solver implementations, per the explicit requirement:
    STRICT   = only VALID_FULL (4/4) scenes
    FLEXIBLE = VALID_FULL + VALID_PARTIAL (>=3/4) scenes
compare_strict_vs_flexible runs both and reports how much the result
actually differs (translation / SO(3) geodesic rotation / residual), so
the impact of including PARTIAL scenes can be judged quantitatively rather
than assumed.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from camera_lidar.pipeline import calibrate_single_scene
from camera_lidar.solver import compute_residuals, solve_rigid_transform
from camera_lidar.types import (
    CameraLidarCalibrationResult,
    CapturedScene,
    FailureReason,
    MultiSceneResult,
    PolicyComparisonResult,
    SceneResidual,
    SceneType,
)
from geometry.transform import invert_transform, rotation_geodesic_distance, to_homogeneous

_DEFAULT_OUTLIER_RMSE_MULTIPLIER = 3.0
_DEFAULT_IMPACT_TRANSLATION_THRESHOLD_M = 0.005   # 5mm
_DEFAULT_IMPACT_ROTATION_THRESHOLD_DEG = 0.5

_POLICY_ALLOWED_TYPES = {
    "strict": frozenset({SceneType.VALID_FULL}),
    "flexible": frozenset({SceneType.VALID_FULL, SceneType.VALID_PARTIAL}),
}


def _detect_all_scenes(included: list[CapturedScene]) -> dict[str, CameraLidarCalibrationResult]:
    """First pass: detect every included scene once, no reference transform
    yet -- needed to know each scene's own classification before a
    reference (built from FULL scenes) can even be computed."""
    return {c.scene_id: calibrate_single_scene(c.scene, roi_mode=c.roi_mode) for c in included}


def _build_reference_transform(
    results: dict[str, CameraLidarCalibrationResult],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """A rough camera-from-lidar estimate from FULL scenes only (never
    ambiguous -- see correspondence.py) used solely to disambiguate PARTIAL
    scenes' correspondence. None if there are no FULL scenes yet."""
    full_pairs = [
        (r.lidar_centers, r.camera_centers) for r in results.values()
        if r.success and r.scene_type == SceneType.VALID_FULL
    ]
    if not full_pairs:
        return None
    ref_lidar = np.concatenate([p[0] for p in full_pairs], axis=0)
    ref_camera = np.concatenate([p[1] for p in full_pairs], axis=0)
    return solve_rigid_transform(ref_lidar, ref_camera)


def calibrate_multi_scene(
    captured_scenes: list[CapturedScene],
    policy: Literal["strict", "flexible"] = "strict",
    outlier_rmse_multiplier: float = _DEFAULT_OUTLIER_RMSE_MULTIPLIER,
) -> MultiSceneResult:
    included = [c for c in captured_scenes if c.included]

    results = _detect_all_scenes(included)

    # PARTIAL scenes' correspondence can be ambiguous without a reference
    # (see correspondence.py) -- re-run just those with a FULL-scene-based
    # reference now that one may be available. FULL scenes are never
    # ambiguous, so re-running them would be redundant work.
    reference_transform = _build_reference_transform(results)
    if reference_transform is not None:
        for captured in included:
            result = results[captured.scene_id]
            if result.success and result.scene_type == SceneType.VALID_PARTIAL:
                results[captured.scene_id] = calibrate_single_scene(
                    captured.scene, roi_mode=captured.roi_mode, reference_transform=reference_transform,
                )

    for captured in included:
        # Keep the Scene Manager table in sync with the latest detection run.
        captured.detection = results[captured.scene_id]

    allowed_types = _POLICY_ALLOWED_TYPES[policy]
    per_scene_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for captured in included:
        result = results[captured.scene_id]
        if not result.success or result.scene_type not in allowed_types:
            continue
        per_scene_pairs.append((captured.scene_id, result.lidar_centers, result.camera_centers))

    if len(per_scene_pairs) < 2:
        return MultiSceneResult(
            success=False,
            failure_reason=FailureReason.NOT_ENOUGH_VALID_SCENES,
            scene_count=len(per_scene_pairs),
            policy=policy,
        )

    all_lidar = np.concatenate([p[1] for p in per_scene_pairs], axis=0)
    all_camera = np.concatenate([p[2] for p in per_scene_pairs], axis=0)

    R_camera_from_lidar, t_camera_from_lidar = solve_rigid_transform(all_lidar, all_camera)
    overall_residuals = compute_residuals(all_lidar, all_camera, R_camera_from_lidar, t_camera_from_lidar)

    T_camera_from_lidar = to_homogeneous(R_camera_from_lidar, t_camera_from_lidar)
    T_lidar_from_camera = invert_transform(T_camera_from_lidar)

    # Re-apply the ONE joint transform to each scene's own points to see
    # how well it fits that scene specifically -- this is what surfaces an
    # outlier scene (its own points don't line up under the shared
    # transform), as distinct from that scene's own single-scene fit.
    per_scene: list[SceneResidual] = []
    for scene_id, lidar_matched, camera_matched in per_scene_pairs:
        scene_residuals = compute_residuals(lidar_matched, camera_matched, R_camera_from_lidar, t_camera_from_lidar)
        per_scene.append(SceneResidual(scene_id=scene_id, rmse_m=scene_residuals.rmse, p95_m=scene_residuals.p95, is_outlier=False))

    rmse_values = np.array([s.rmse_m for s in per_scene])
    median_rmse = float(np.median(rmse_values))
    threshold = median_rmse * outlier_rmse_multiplier
    outlier_ids: list[str] = []
    for s in per_scene:
        if s.rmse_m > threshold > 0.0 or (median_rmse == 0.0 and s.rmse_m > 0.0):
            s.is_outlier = True
            outlier_ids.append(s.scene_id)

    return MultiSceneResult(
        success=True,
        R_camera_from_lidar=R_camera_from_lidar,
        t_camera_from_lidar=t_camera_from_lidar,
        T_camera_from_lidar=T_camera_from_lidar,
        T_lidar_from_camera=T_lidar_from_camera,
        residual_rmse_m=overall_residuals.rmse,
        residual_mean_m=overall_residuals.mean,
        residual_median_m=overall_residuals.median,
        residual_p95_m=overall_residuals.p95,
        residual_max_m=overall_residuals.max,
        scene_count=len(per_scene_pairs),
        per_scene=per_scene,
        outlier_scene_ids=outlier_ids,
        policy=policy,
    )


def compare_strict_vs_flexible(
    captured_scenes: list[CapturedScene],
    outlier_rmse_multiplier: float = _DEFAULT_OUTLIER_RMSE_MULTIPLIER,
    impact_translation_threshold_m: float = _DEFAULT_IMPACT_TRANSLATION_THRESHOLD_M,
    impact_rotation_threshold_deg: float = _DEFAULT_IMPACT_ROTATION_THRESHOLD_DEG,
) -> PolicyComparisonResult:
    """Runs calibrate_multi_scene twice (STRICT, FLEXIBLE -- same solver,
    different scene subset) and reports how much PARTIAL scenes actually
    change the result, per the explicit "quantify the impact, don't just
    assume" requirement."""
    strict_result = calibrate_multi_scene(captured_scenes, policy="strict", outlier_rmse_multiplier=outlier_rmse_multiplier)
    flexible_result = calibrate_multi_scene(captured_scenes, policy="flexible", outlier_rmse_multiplier=outlier_rmse_multiplier)

    if not strict_result.success or not flexible_result.success:
        return PolicyComparisonResult(strict_result=strict_result, flexible_result=flexible_result)

    translation_diff = float(np.linalg.norm(strict_result.t_camera_from_lidar - flexible_result.t_camera_from_lidar))
    rotation_diff = rotation_geodesic_distance(
        strict_result.R_camera_from_lidar, flexible_result.R_camera_from_lidar, degrees=True,
    )
    residual_diff = abs(strict_result.residual_rmse_m - flexible_result.residual_rmse_m)

    impact = "HIGH" if (
        translation_diff > impact_translation_threshold_m or rotation_diff > impact_rotation_threshold_deg
    ) else "LOW"

    return PolicyComparisonResult(
        strict_result=strict_result,
        flexible_result=flexible_result,
        translation_difference_m=translation_diff,
        rotation_difference_deg=rotation_diff,
        residual_difference_m=residual_diff,
        impact=impact,
    )
