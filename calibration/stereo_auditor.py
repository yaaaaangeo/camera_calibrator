"""
Evidence-oriented stereo calibration quality checks.

These helpers intentionally avoid claiming absolute accuracy without ground truth.
They combine dataset coverage, sync health, residuals, and geometry consistency into
a practical GT-free confidence report for camera-to-camera calibration.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from calibration.stereo import StereoCalibrationResult, StereoPairObservation, stats_from_values


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _stats_dict(values: list[float]) -> dict[str, float | None]:
    stats = stats_from_values(_finite(values))
    return {
        "mean": stats.mean,
        "median": stats.median,
        "rmse": stats.rmse,
        "p95": stats.p95,
        "p99": stats.p99,
        "max": stats.max,
    }


def compute_sync_guard(
    pairs: list[StereoPairObservation],
    *,
    threshold_ms: float = 30.0,
) -> dict[str, Any]:
    deltas = [abs(float(p.sync_delta_ms)) for p in pairs if p.sync_delta_ms is not None]
    suspect_pairs = [
        p.pair_id for p in pairs
        if p.sync_delta_ms is not None and abs(float(p.sync_delta_ms)) > threshold_ms
    ]
    stats = _stats_dict(deltas)
    jitter = float(np.std(deltas)) if deltas else None
    drift = None
    if len(deltas) >= 2:
        drift = float((deltas[-1] - deltas[0]) / max(len(deltas) - 1, 1))
    p95 = stats["p95"]
    status = "UNKNOWN"
    if deltas:
        status = "GOOD" if p95 is not None and p95 <= threshold_ms else "SYNC SUSPECT"
    return {
        "mode": "timestamp_delta",
        "threshold_ms": threshold_ms,
        "pair_count_with_timestamp": len(deltas),
        "missing_timestamp_pairs": len(pairs) - len(deltas),
        "timestamp_delta_ms": stats,
        "jitter_ms": jitter,
        "clock_drift_ms_per_pair": drift,
        "suspect_pair_ids": suspect_pairs,
        "status": status,
        "warnings": (
            [f"{len(suspect_pairs)} pair(s) exceed {threshold_ms:.1f} ms sync threshold."]
            if suspect_pairs else []
        ),
    }


def _coverage_grid(points: list[np.ndarray], image_size: tuple[int, int], rows: int = 3, cols: int = 4) -> dict[str, Any]:
    w, h = image_size
    grid = np.zeros((rows, cols), dtype=int)
    for pts in points:
        flat = np.asarray(pts, dtype=float).reshape(-1, 2)
        for x, y in flat:
            if w <= 0 or h <= 0:
                continue
            col = min(cols - 1, max(0, int(x / max(w, 1) * cols)))
            row = min(rows - 1, max(0, int(y / max(h, 1) * rows)))
            grid[row, col] += 1
    occupied = int(np.count_nonzero(grid))
    score = 100.0 * occupied / float(rows * cols)
    return {
        "rows": rows,
        "cols": cols,
        "occupied_cells": occupied,
        "total_cells": rows * cols,
        "score": score,
        "grid": grid.tolist(),
    }


def _weakest_grid_hint(grid: dict[str, Any]) -> str | None:
    values = np.asarray(grid.get("grid", []), dtype=float)
    if values.size == 0:
        return None
    row, col = np.unravel_index(int(np.argmin(values)), values.shape)
    vertical = ["TOP", "CENTER", "BOTTOM"][min(row, 2)]
    horizontal = ["LEFT", "CENTER", "RIGHT", "RIGHT"][min(col, 3)]
    if vertical == "CENTER" and horizontal == "CENTER":
        return "Move board through center at different depths"
    return f"Move board toward {vertical}-{horizontal}"


def compute_capture_coach(
    pairs: list[StereoPairObservation],
    image_size: tuple[int, int],
    *,
    target_pairs: int = 50,
) -> dict[str, Any]:
    used = [p for p in pairs if p.used]
    common_counts = [float(p.common_count) for p in used]
    cam1_points = [p.image_points_cam1 for p in used]
    cam2_points = [p.image_points_cam2 for p in used]
    cam1_coverage = _coverage_grid(cam1_points, image_size)
    cam2_coverage = _coverage_grid(cam2_points, image_size)
    joint_score = min(cam1_coverage["score"], cam2_coverage["score"])
    quality_scores = [float(p.quality_score) for p in used if p.quality_score > 0.0]
    pose_scores = [
        float(p.quality_components.get("pose_diversity", 0.0))
        for p in used if p.quality_components
    ]
    sync_scores = [
        float(p.quality_components.get("timestamp_sync", 0.0))
        for p in used if p.quality_components
    ]
    dataset_score = float(np.mean([
        min(100.0, len(used) / target_pairs * 100.0),
        joint_score,
        np.mean(quality_scores) if quality_scores else 0.0,
        np.mean(pose_scores) if pose_scores else 0.0,
    ]))
    recommendations: list[str] = []
    if len(used) < target_pairs:
        recommendations.append(f"Collect at least {target_pairs} usable pairs; current usable pairs: {len(used)}.")
    hint1 = _weakest_grid_hint(cam1_coverage)
    hint2 = _weakest_grid_hint(cam2_coverage)
    if joint_score < 70.0 and hint1:
        recommendations.append(hint1)
    if hint2 and hint2 != hint1 and joint_score < 70.0:
        recommendations.append(f"Camera 2 also needs: {hint2}")
    if pose_scores and float(np.mean(pose_scores)) < 55.0:
        recommendations.append("Increase pose novelty: add yaw, pitch, roll, and depth variation.")
    if sync_scores and float(np.mean(sync_scores)) < 70.0:
        recommendations.append("Improve sync before adding more pairs.")
    return {
        "target_pairs": target_pairs,
        "usable_pairs": len(used),
        "rejected_pairs": len([p for p in pairs if not p.used]),
        "common_corners": _stats_dict(common_counts),
        "camera1_coverage": cam1_coverage,
        "camera2_coverage": cam2_coverage,
        "joint_coverage_score": joint_score,
        "average_pair_quality": float(np.mean(quality_scores)) if quality_scores else None,
        "average_pose_diversity": float(np.mean(pose_scores)) if pose_scores else None,
        "dataset_quality_score": dataset_score,
        "dataset_ready": len(used) >= target_pairs and dataset_score >= 70.0,
        "recommendations": recommendations,
    }


def compute_pose_consistency(
    result: StereoCalibrationResult,
    pairs: list[StereoPairObservation],
) -> dict[str, Any]:
    trans_errors_mm: list[float] = []
    rot_errors_deg: list[float] = []
    K1, D1 = result.camera1.camera_matrix, result.camera1.distortion
    K2, D2 = result.camera2.camera_matrix, result.camera2.distortion
    R12 = result.R_cam2_from_cam1
    t12 = result.t_cam2_from_cam1.reshape(3, 1)
    for pair in pairs:
        if pair.common_count < 4:
            continue
        ok1, rvec1, tvec1 = cv2.solvePnP(pair.object_points, pair.image_points_cam1, K1, D1, flags=cv2.SOLVEPNP_ITERATIVE)
        ok2, rvec2, tvec2 = cv2.solvePnP(pair.object_points, pair.image_points_cam2, K2, D2, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok1 or not ok2:
            continue
        R_board_cam1, _ = cv2.Rodrigues(rvec1)
        R_board_cam2, _ = cv2.Rodrigues(rvec2)
        R_pred = R12 @ R_board_cam1
        t_pred = R12 @ tvec1.reshape(3, 1) + t12
        trans_errors_mm.append(float(np.linalg.norm(t_pred - tvec2.reshape(3, 1)) * 1000.0))
        R_delta = R_pred @ R_board_cam2.T
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R_delta)) - 1.0) / 2.0))))
        rot_errors_deg.append(angle)
    return {
        "translation_error_mm": _stats_dict(trans_errors_mm),
        "rotation_error_deg": _stats_dict(rot_errors_deg),
        "evaluated_pairs": len(trans_errors_mm),
    }


def compute_reconstruction_check(
    result: StereoCalibrationResult,
    pairs: list[StereoPairObservation],
) -> dict[str, Any]:
    distance_errors_mm: list[float] = []
    plane_errors_mm: list[float] = []
    scale_errors_percent: list[float] = []
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([result.R_cam2_from_cam1, result.t_cam2_from_cam1.reshape(3, 1)])
    for pair in pairs:
        if pair.common_count < 4:
            continue
        pts1 = cv2.undistortPoints(pair.image_points_cam1, result.camera1.camera_matrix, result.camera1.distortion).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pair.image_points_cam2, result.camera2.camera_matrix, result.camera2.distortion).reshape(-1, 2)
        homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        denom = homog[3:].copy()
        denom[np.abs(denom) < 1e-12] = 1e-12
        xyz = (homog[:3] / denom).T
        ok, rvec, tvec = cv2.solvePnP(pair.object_points, pair.image_points_cam1, result.camera1.camera_matrix, result.camera1.distortion)
        if ok:
            R_board_cam1, _ = cv2.Rodrigues(rvec)
            expected = (R_board_cam1 @ pair.object_points.reshape(-1, 3).T + tvec.reshape(3, 1)).T
            distance_errors_mm.extend((np.linalg.norm(xyz - expected, axis=1) * 1000.0).tolist())
        if xyz.shape[0] >= 4:
            centered = xyz - np.mean(xyz, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            dist = centered @ normal
            plane_errors_mm.extend((np.abs(dist) * 1000.0).tolist())
            obj = pair.object_points.reshape(-1, 3).astype(np.float64)
            for i in range(obj.shape[0]):
                distances = np.linalg.norm(obj - obj[i], axis=1)
                candidates = np.where(distances > 1e-9)[0]
                if candidates.size == 0:
                    continue
                j = int(candidates[np.argmin(distances[candidates])])
                expected = float(np.linalg.norm(obj[i] - obj[j]))
                measured = float(np.linalg.norm(xyz[i] - xyz[j]))
                if expected > 1e-9 and math.isfinite(measured):
                    scale_errors_percent.append(abs(measured - expected) / expected * 100.0)
    return {
        "point_to_pose_error_mm": _stats_dict(distance_errors_mm),
        "plane_error_mm": _stats_dict(plane_errors_mm),
        "local_board_scale_error_percent": _stats_dict(scale_errors_percent),
        "evaluated_points": len(distance_errors_mm),
    }


def compute_bootstrap_uncertainty(
    result: StereoCalibrationResult,
    pairs: list[StereoPairObservation],
    *,
    samples: int = 12,
    seed: int = 7,
) -> dict[str, Any]:
    usable = [p for p in pairs if p.used and p.common_count >= 4]
    if len(usable) < 4:
        return compute_uncertainty_proxy(result)
    from calibration.stereo import calibrate_stereo

    rng = np.random.default_rng(seed)
    baselines: list[float] = []
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    subset_size = max(2, int(round(len(usable) * 0.8)))
    for _ in range(samples):
        indices = rng.choice(len(usable), size=subset_size, replace=True)
        subset = [usable[int(i)] for i in indices]
        try:
            boot = calibrate_stereo(
                subset,
                result.camera1,
                result.camera2,
                result.image_size,
                fix_intrinsics=True,
                compute_holdout=False,
                compute_audit=False,
            )
        except Exception:
            continue
        baselines.append(float(boot.baseline * 1000.0))
        translations.append(boot.t_cam2_from_cam1.reshape(3).astype(np.float64) * 1000.0)
        rvec, _ = cv2.Rodrigues(boot.R_cam2_from_cam1)
        rotations.append(np.degrees(rvec.reshape(3)))
    if len(baselines) < 3:
        fallback = compute_uncertainty_proxy(result)
        fallback["bootstrap_successful_samples"] = len(baselines)
        return fallback
    b = np.asarray(baselines, dtype=np.float64)
    t = np.asarray(translations, dtype=np.float64)
    r = np.asarray(rotations, dtype=np.float64)
    return {
        "method": "bootstrap over stereo pairs",
        "bootstrap_successful_samples": len(baselines),
        "baseline_mean_mm": float(np.mean(b)),
        "baseline_std_mm": float(np.std(b, ddof=1)),
        "baseline_95ci_mm": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
        "translation_std_mm_xyz": np.std(t, axis=0, ddof=1).tolist(),
        "rotation_std_deg_rodrigues": np.std(r, axis=0, ddof=1).tolist(),
        "note": "Bootstrap shows repeatability under pair resampling; external ground truth is still required for absolute accuracy.",
    }


def compute_uncertainty_proxy(result: StereoCalibrationResult) -> dict[str, Any]:
    baseline_mm = float(result.baseline * 1000.0)
    residual = result.holdout_validation_error.rmse if result.holdout_validation_error else result.epipolar_error.rmse
    residual = float(residual or 0.0)
    baseline_std_mm = max(0.001, residual * max(baseline_mm, 1.0) * 0.001)
    return {
        "method": "GT-free proxy from validation residuals",
        "baseline_mean_mm": baseline_mm,
        "baseline_std_mm": baseline_std_mm,
        "baseline_95ci_mm": [baseline_mm - 1.96 * baseline_std_mm, baseline_mm + 1.96 * baseline_std_mm],
        "note": "Use repeat captures or external ground truth for absolute uncertainty.",
    }


def build_stereo_evidence_report(
    result: StereoCalibrationResult,
    pairs: list[StereoPairObservation],
    image_size: tuple[int, int],
    *,
    full_uncertainty: bool = True,
) -> dict[str, Any]:
    capture = compute_capture_coach(pairs, image_size)
    sync = compute_sync_guard(pairs)
    pose = compute_pose_consistency(result, pairs)
    reconstruction = compute_reconstruction_check(result, pairs)
    uncertainty = compute_bootstrap_uncertainty(result, pairs) if full_uncertainty else compute_uncertainty_proxy(result)
    audit = {
        "stereo_rms_px": result.stereo_rms,
        "epipolar_error_px": {
            "mean": result.epipolar_error.mean,
            "p95": result.epipolar_error.p95,
            "max": result.epipolar_error.max,
        },
        "sampson_distance": {
            "mean": result.sampson_distance.mean,
            "p95": result.sampson_distance.p95,
        },
        "rectification_vertical_error_px": {
            "rmse": result.rectification_vertical_error.rmse,
            "p95": result.rectification_vertical_error.p95,
            "max": result.rectification_vertical_error.max,
        },
        "holdout": {
            "train_pairs": result.holdout_train_pair_count,
            "validation_pairs": result.holdout_validation_pair_count,
            "generalization_gap_px": result.holdout_generalization_gap,
        },
        "cross_camera_pose_consistency": pose,
        "reconstruction": reconstruction,
        "stability_uncertainty": uncertainty,
    }
    checks = [
        capture["dataset_quality_score"] >= 70.0,
        sync["status"] in {"GOOD", "UNKNOWN"},
        result.epipolar_error.p95 is not None and result.epipolar_error.p95 <= 1.0,
        result.rectification_vertical_error.p95 is not None and result.rectification_vertical_error.p95 <= 1.0,
    ]
    pose_p95 = pose["translation_error_mm"]["p95"]
    if pose_p95 is not None:
        checks.append(pose_p95 <= 10.0)
    scale_p95 = reconstruction["local_board_scale_error_percent"]["p95"]
    if scale_p95 is not None:
        checks.append(scale_p95 <= 2.0)
    passed = sum(1 for c in checks if c)
    confidence = "HIGH" if passed >= len(checks) - 1 else "MEDIUM" if passed >= max(2, len(checks) // 2) else "LOW"
    warnings: list[str] = []
    warnings.extend(capture.get("recommendations", []))
    warnings.extend(sync.get("warnings", []))
    if confidence != "HIGH":
        warnings.append("GT-free evidence is not strong enough for a high-confidence result.")
    report = {
        "title": "Camera-to-Camera Calibration Evidence Report",
        "confidence": confidence,
        "evidence_model": "GT-free multi-evidence validation",
        "passed_checks": passed,
        "total_checks": len(checks),
        "warnings": warnings,
        "absolute_accuracy_claim": "Not available without external ground truth.",
    }
    return {
        "capture_coach": capture,
        "sync_guard": sync,
        "calibration_audit": audit,
        "evidence_report": report,
    }
