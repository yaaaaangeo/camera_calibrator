"""
camera_calibrator.camera_lidar.pipeline
===========================================

Single-scene FAST-Calib orchestration:

    CalibrationScene
        -> detect camera target (camera_detector)
        -> detect lidar target (lidar_detector)
        -> canonical common-feature correspondence (correspondence)
        -> classify FULL / PARTIAL / INVALID (SceneType)
        -> solve rigid transform (reuses the correspondence's own fit)
        -> CameraLidarCalibrationResult

Each stage short-circuits with a specific FailureReason on failure, so
callers can show *why* calibration failed instead of a bare error.

Common-feature count is the INTERSECTION of what camera and LiDAR each
independently, canonically identified (never either sensor's raw count
alone, and never detection order) -- see correspondence.match_partial_centers
for how that intersection and its corner identities get resolved, including
the "why PARTIAL correspondence can be ambiguous" limitation.

This module (and everything camera_lidar/ imports) is ROS-independent --
see the dependency-direction note in camera_lidar/types.py. Callers (UI
workers, future bag/live adapters) are responsible for producing a
CalibrationScene from whatever source they read.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from geometry.transform import invert_transform, to_homogeneous

from camera_lidar.camera_detector import detect_camera_target
from camera_lidar.correspondence import match_partial_centers
from camera_lidar.lidar_detector import detect_lidar_target, detect_lidar_target_auto
from camera_lidar.solver import compute_residuals
from camera_lidar.target_config import CORNER_ORDER
from camera_lidar.types import CalibrationScene, CameraLidarCalibrationResult, FailureReason, SceneType

_ALL_CORNER_IDS = frozenset(CORNER_ORDER)
_MIN_COMMON_FOR_USABLE_SCENE = 3


def calibrate_single_scene(
    scene: CalibrationScene,
    roi_mode: str = "manual",
    reference_transform: Optional[tuple[np.ndarray, np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CameraLidarCalibrationResult:
    """roi_mode: "manual" uses scene.roi (a caller-set box, via
    lidar_detector.detect_lidar_target); "auto" ignores scene.roi and runs
    the LiDAR-only multi-plane search (lidar_detector.detect_lidar_target_auto)
    instead. camera_lidar.multi_scene reuses this exact function per scene
    so a scene's single-scene preview and its contribution to the joint
    multi-scene solve are always computed identically.

    reference_transform: optional (R_camera_from_lidar, t_camera_from_lidar)
    -- typically the current best FULL-scene-only estimate -- used only to
    break PARTIAL-scene correspondence ties (see correspondence.py). Has no
    effect on FULL (4/4) scenes."""
    camera_result = detect_camera_target(scene.image.image, scene.intrinsics, scene.target)
    if not camera_result.success:
        return CameraLidarCalibrationResult(
            success=False, failure_reason=camera_result.failure_reason, camera_detection=camera_result,
        )

    if cancel_check is not None and cancel_check():
        return CameraLidarCalibrationResult(
            success=False, failure_reason=FailureReason.CANCELLED, camera_detection=camera_result,
        )

    if roi_mode == "auto":
        lidar_result = detect_lidar_target_auto(scene.cloud, scene.target, cancel_check=cancel_check)
    else:
        lidar_result = detect_lidar_target(scene.cloud, scene.roi, scene.target, cancel_check=cancel_check)
    if not lidar_result.success:
        return CameraLidarCalibrationResult(
            success=False, failure_reason=lidar_result.failure_reason,
            camera_detection=camera_result, lidar_detection=lidar_result,
        )

    camera_ids_to_centers = {
        cid: camera_result.circle_centers[i]
        for i, cid in enumerate(CORNER_ORDER)
        if cid in camera_result.detected_ids
    }
    correspondence = match_partial_centers(
        camera_ids_to_centers, lidar_result.circle_centers, reference_transform=reference_transform,
    )
    if correspondence is None or len(correspondence.common_ids) < _MIN_COMMON_FOR_USABLE_SCENE:
        common_ids = correspondence.common_ids if correspondence is not None else frozenset()
        return CameraLidarCalibrationResult(
            success=False, failure_reason=FailureReason.INSUFFICIENT_COMMON_FEATURES,
            camera_detection=camera_result, lidar_detection=lidar_result,
            scene_type=SceneType.INVALID, common_ids=common_ids,
            missing_from_camera=_ALL_CORNER_IDS - camera_result.detected_ids,
            missing_from_lidar=camera_result.detected_ids - common_ids,
        )

    R_camera_from_lidar = correspondence.R_camera_from_lidar
    t_camera_from_lidar = correspondence.t_camera_from_lidar
    residuals = compute_residuals(
        correspondence.lidar_centers_matched, correspondence.camera_centers,
        R_camera_from_lidar, t_camera_from_lidar,
    )

    T_camera_from_lidar = to_homogeneous(R_camera_from_lidar, t_camera_from_lidar)
    T_lidar_from_camera = invert_transform(T_camera_from_lidar)

    scene_type = SceneType.VALID_FULL if len(correspondence.common_ids) == 4 else SceneType.VALID_PARTIAL

    return CameraLidarCalibrationResult(
        success=True,
        R_camera_from_lidar=R_camera_from_lidar,
        t_camera_from_lidar=t_camera_from_lidar,
        T_camera_from_lidar=T_camera_from_lidar,
        T_lidar_from_camera=T_lidar_from_camera,
        camera_centers=correspondence.camera_centers,
        lidar_centers=correspondence.lidar_centers_matched,
        residual_rmse_m=residuals.rmse,
        residual_mean_m=residuals.mean,
        residual_median_m=residuals.median,
        residual_p95_m=residuals.p95,
        residual_max_m=residuals.max,
        camera_detection=camera_result,
        lidar_detection=lidar_result,
        scene_type=scene_type,
        common_ids=correspondence.common_ids,
        missing_from_camera=_ALL_CORNER_IDS - camera_result.detected_ids,
        missing_from_lidar=camera_result.detected_ids - correspondence.common_ids,
        correspondence_ambiguous=correspondence.ambiguous,
    )
