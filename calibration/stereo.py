"""
camera_calibrator.calibration.stereo
====================================

Camera-to-camera stereo extrinsic calibration backend.

The convention used throughout this module is:

    P_cam2 = R_cam2_from_cam1 @ P_cam1 + t_cam2_from_cam1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from calibration.calibration_io import StandardCalibration
from calibration.types import CameraModelType, Dataset, DetectionResult


@dataclass
class StereoPairObservation:
    pair_id: str
    object_points: np.ndarray
    image_points_cam1: np.ndarray
    image_points_cam2: np.ndarray
    common_ids: np.ndarray
    detected_points_cam1: np.ndarray | None = None
    detected_points_cam2: np.ndarray | None = None
    detected_ids_cam1: np.ndarray | None = None
    detected_ids_cam2: np.ndarray | None = None
    timestamp_cam1: float | None = None
    timestamp_cam2: float | None = None
    image_path_cam1: str | None = None
    image_path_cam2: str | None = None
    sync_delta_ms: float | None = None
    used: bool = True
    rejected_reason: str | None = None
    quality_score: float = 0.0
    quality_status: str = "Unknown"
    quality_components: dict[str, float] = field(default_factory=dict)
    quality_warnings: list[str] = field(default_factory=list)

    @property
    def common_count(self) -> int:
        return int(self.common_ids.size)


@dataclass
class StereoValidationStats:
    mean: float | None = None
    median: float | None = None
    rmse: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None


@dataclass
class StereoPairValidation:
    pair_id: str
    common_corners: int
    epipolar_mean: float | None = None
    sampson_mean: float | None = None
    vertical_mean: float | None = None
    status: str = "Good"


@dataclass
class StereoPathPairingResult:
    camera1_paths: list[str]
    camera2_paths: list[str]
    warnings: list[str] = field(default_factory=list)
    unmatched_camera1_paths: list[str] = field(default_factory=list)
    unmatched_camera2_paths: list[str] = field(default_factory=list)

    @property
    def pair_count(self) -> int:
        return min(len(self.camera1_paths), len(self.camera2_paths))


@dataclass
class StereoCalibrationResult:
    camera1: StandardCalibration
    camera2: StandardCalibration
    image_size: tuple[int, int]
    stereo_rms: float
    R_cam2_from_cam1: np.ndarray
    t_cam2_from_cam1: np.ndarray
    E: np.ndarray
    F: np.ndarray
    T_cam2_from_cam1: np.ndarray
    R_cam1_from_cam2: np.ndarray
    t_cam1_from_cam2: np.ndarray
    T_cam1_from_cam2: np.ndarray
    baseline: float
    roll_pitch_yaw_deg: tuple[float, float, float]
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    epipolar_error: StereoValidationStats = field(default_factory=StereoValidationStats)
    sampson_distance: StereoValidationStats = field(default_factory=StereoValidationStats)
    rectification_vertical_error: StereoValidationStats = field(default_factory=StereoValidationStats)
    pair_validations: list[StereoPairValidation] = field(default_factory=list)
    holdout_training_error: StereoValidationStats | None = None
    holdout_validation_error: StereoValidationStats | None = None
    holdout_generalization_gap: float | None = None
    holdout_train_pair_count: int = 0
    holdout_validation_pair_count: int = 0
    used_pair_count: int = 0
    rejected_pair_count: int = 0
    total_common_corners: int = 0
    capture_coach: dict[str, Any] = field(default_factory=dict)
    sync_guard: dict[str, Any] = field(default_factory=dict)
    calibration_audit: dict[str, Any] = field(default_factory=dict)
    evidence_report: dict[str, Any] = field(default_factory=dict)


def _as_point_array(points: np.ndarray, dims: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, dims)
    return arr


def _flat_ids(ids: np.ndarray | None) -> np.ndarray:
    if ids is None:
        return np.array([], dtype=np.int32)
    return np.asarray(ids, dtype=np.int32).reshape(-1)


def extract_timestamp_from_filename(path: str) -> float | None:
    """Extract a sortable timestamp-like value from a filename.

    Supports common names such as ``cam_1692600000123.png`` and
    ``frame_12.345.jpg``. Values with 13+ digits are treated as milliseconds.
    """
    stem = Path(path).stem
    matches = re.findall(r"\d+(?:\.\d+)?", stem)
    if not matches:
        return None
    value = float(matches[-1])
    if value > 1e12:
        value /= 1000.0
    return value


def extract_timestamp_from_exif(path: str) -> float | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get(36867) or exif.get(306)
    except Exception:
        return None
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def extract_timestamp_from_sidecar(path: str) -> float | None:
    candidates = [Path(path).with_suffix(".json"), Path(f"{path}.json")]
    keys = ("timestamp", "timestamp_sec", "stamp", "time", "header_stamp", "ros_time")
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value) / 1000.0 if float(value) > 1e12 else float(value)
            if isinstance(value, dict):
                sec = value.get("sec", value.get("secs"))
                nsec = value.get("nanosec", value.get("nsec", value.get("nanoseconds", 0)))
                if sec is not None:
                    return float(sec) + float(nsec or 0) / 1e9
    return None


def _timestamp_for_pairing(path: str, mode: str) -> float | None:
    if mode == "exif":
        return extract_timestamp_from_exif(path)
    if mode in {"ros_timestamp", "sidecar"}:
        return extract_timestamp_from_sidecar(path)
    return extract_timestamp_from_filename(path)


def pair_image_paths(
    camera1_paths: list[str],
    camera2_paths: list[str],
    *,
    mode: str = "sorted",
    max_timestamp_delta_ms: float = 30.0,
) -> StereoPathPairingResult:
    warnings: list[str] = []
    if mode == "stem":
        by_stem2 = {Path(p).stem: p for p in camera2_paths}
        p1: list[str] = []
        p2: list[str] = []
        for path1 in camera1_paths:
            matched = by_stem2.get(Path(path1).stem)
            if matched is not None:
                p1.append(path1)
                p2.append(matched)
        if len(p1) != len(camera1_paths) or len(p2) != len(camera2_paths):
            warnings.append(f"Stem matching paired {len(p1)} files; unmatched files were ignored.")
        return StereoPathPairingResult(p1, p2, warnings)

    if mode in {"timestamp", "exif", "ros_timestamp", "sidecar"}:
        stamped1 = [(p, _timestamp_for_pairing(p, mode)) for p in camera1_paths]
        stamped2 = [(p, _timestamp_for_pairing(p, mode)) for p in camera2_paths]
        usable1 = [(p, t) for p, t in stamped1 if t is not None]
        usable2 = sorted([(p, t) for p, t in stamped2 if t is not None], key=lambda item: item[1])
        if len(usable1) != len(camera1_paths) or len(usable2) != len(camera2_paths):
            warnings.append(f"{mode} matching ignored files without usable timestamps.")
        max_delta = max_timestamp_delta_ms / 1000.0
        p1: list[str] = []
        p2: list[str] = []
        remaining2 = usable2.copy()
        unmatched1: list[str] = []
        for path1, ts1 in sorted(usable1, key=lambda item: item[1]):
            if not remaining2:
                unmatched1.append(path1)
                break
            best_index, best = min(enumerate(remaining2), key=lambda item: abs(item[1][1] - ts1))
            if abs(best[1] - ts1) <= max_delta:
                p1.append(path1)
                p2.append(best[0])
                remaining2.pop(best_index)
            else:
                unmatched1.append(path1)
        if len(p1) != min(len(usable1), len(usable2)):
            warnings.append(f"{mode} matching paired {len(p1)} files within {max_timestamp_delta_ms:.1f} ms.")
        return StereoPathPairingResult(
            p1,
            p2,
            warnings,
            unmatched_camera1_paths=unmatched1 + [p for p, t in stamped1 if t is None],
            unmatched_camera2_paths=[p for p, _t in remaining2] + [p for p, t in stamped2 if t is None],
        )

    count = min(len(camera1_paths), len(camera2_paths))
    if len(camera1_paths) != len(camera2_paths):
        warnings.append(f"Sorted matching uses first {count} files; folder counts differ.")
    return StereoPathPairingResult(camera1_paths[:count], camera2_paths[:count], warnings)


def _average_optional(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def score_stereo_pair_quality(
    detection_cam1: DetectionResult,
    detection_cam2: DetectionResult,
    *,
    common_corners: int,
    min_common_corners: int = 6,
    sync_delta_ms: float | None = None,
    sync_warning_ms: float = 30.0,
) -> tuple[float, str, list[str]]:
    warnings: list[str] = []
    corner_score = min(common_corners / max(min_common_corners * 2.0, 1.0), 1.0) * 35.0
    if common_corners < min_common_corners:
        warnings.append(f"Only {common_corners} common corners (< {min_common_corners}).")

    if sync_delta_ms is None:
        sync_score = 10.0
    elif sync_delta_ms <= sync_warning_ms:
        sync_score = 20.0
    else:
        sync_score = max(0.0, 20.0 * (1.0 - (sync_delta_ms - sync_warning_ms) / max(sync_warning_ms, 1.0)))
        warnings.append(f"Sync delta {sync_delta_ms:.1f} ms exceeds {sync_warning_ms:.1f} ms.")

    confidence = _average_optional([detection_cam1.corner_confidence, detection_cam2.corner_confidence])
    confidence_score = 15.0 if confidence is None else max(0.0, min(confidence, 1.0)) * 25.0
    if confidence is not None and confidence < 0.55:
        warnings.append(f"Low corner confidence ({confidence:.2f}).")

    area = _average_optional([detection_cam1.board_area_ratio, detection_cam2.board_area_ratio])
    area_score = 10.0 if area is None else min(max(area / 0.12, 0.0), 1.0) * 20.0
    if area is not None and area < 0.04:
        warnings.append(f"Small board coverage ({area * 100.0:.1f}%).")

    score = float(np.clip(corner_score + sync_score + confidence_score + area_score, 0.0, 100.0))
    if common_corners < min_common_corners:
        status = "Reject"
    elif score >= 75.0:
        status = "Good"
    elif score >= 50.0:
        status = "Warning"
    else:
        status = "Reject"
    return score, status, warnings


def _board_position_score(
    detection: DetectionResult,
    image_size: tuple[int, int] | None,
) -> float | None:
    if image_size is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    if detection.board_center_px is not None:
        cx, cy = detection.board_center_px
    elif detection.corners is not None:
        pts = np.asarray(detection.corners, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return None
        cx, cy = np.mean(pts, axis=0)
    else:
        return None

    nx = abs(float(cx) - width / 2.0) / max(width / 2.0, 1.0)
    ny = abs(float(cy) - height / 2.0) / max(height / 2.0, 1.0)
    normalized_distance = min(1.0, math.hypot(nx, ny) / math.sqrt(2.0))
    return float(max(0.0, min(100.0, (1.0 - normalized_distance) * 100.0)))


def stereo_pair_quality_components(
    detection_cam1: DetectionResult,
    detection_cam2: DetectionResult,
    *,
    common_corners: int,
    min_common_corners: int = 6,
    sync_delta_ms: float | None = None,
    sync_warning_ms: float = 30.0,
    image_size_cam1: tuple[int, int] | None = None,
    image_size_cam2: tuple[int, int] | None = None,
) -> dict[str, float]:
    confidence = _average_optional([detection_cam1.corner_confidence, detection_cam2.corner_confidence])
    area = _average_optional([detection_cam1.board_area_ratio, detection_cam2.board_area_ratio])
    position = _average_optional([
        _board_position_score(detection_cam1, image_size_cam1),
        _board_position_score(detection_cam2, image_size_cam2),
    ])
    sync_score = 50.0 if sync_delta_ms is None else max(0.0, min(100.0, 100.0 * (1.0 - sync_delta_ms / max(sync_warning_ms * 2.0, 1.0))))
    size_score = 50.0 if area is None else max(0.0, min(100.0, area / 0.12 * 100.0))
    return {
        "common_corners": max(0.0, min(100.0, common_corners / max(min_common_corners * 2.0, 1.0) * 100.0)),
        "board_size": size_score,
        "timestamp_sync": sync_score,
        "detection_confidence": 60.0 if confidence is None else max(0.0, min(100.0, confidence * 100.0)),
        "board_position": 50.0 if position is None else position,
        "pose_diversity": 0.0,
    }


def mark_stereo_outlier_candidates(
    result: StereoCalibrationResult,
    *,
    epipolar_threshold_px: float = 1.0,
    vertical_threshold_px: float = 1.0,
) -> list[str]:
    outliers: list[str] = []
    for row in result.pair_validations:
        is_outlier = (
            (row.epipolar_mean is not None and row.epipolar_mean > epipolar_threshold_px)
            or (row.vertical_mean is not None and row.vertical_mean > vertical_threshold_px)
        )
        row.status = "Outlier" if is_outlier else "Good"
        if is_outlier:
            outliers.append(row.pair_id)
    return outliers


def reject_pairs_by_id(
    pairs: list[StereoPairObservation],
    pair_ids: set[str],
    *,
    reason: str = "Rejected as stereo outlier candidate",
) -> int:
    count = 0
    for pair in pairs:
        if pair.pair_id in pair_ids and pair.used:
            set_pair_used(pair, False, reason)
            count += 1
    return count


def apply_pair_pose_diversity_scores(pairs: list[StereoPairObservation]) -> None:
    centers = []
    for pair in pairs:
        pts = np.asarray(pair.image_points_cam1, dtype=np.float64).reshape(-1, 2)
        if pts.size:
            centers.append(np.mean(pts, axis=0))
    if len(centers) < 2:
        return
    arr = np.asarray(centers, dtype=np.float64)
    span = np.ptp(arr, axis=0)
    max_span = float(max(np.max(span), 1.0))
    for pair, center in zip(pairs, arr):
        d = float(np.linalg.norm(center - np.mean(arr, axis=0)))
        score = max(0.0, min(100.0, d / max_span * 100.0))
        pair.quality_components["pose_diversity"] = score
        if score < 10.0 and "Low pose diversity." not in pair.quality_warnings:
            pair.quality_warnings.append("Low pose diversity.")


def apply_pair_pose_diversity_scores_from_intrinsics(
    pairs: list[StereoPairObservation],
    camera: StandardCalibration,
) -> None:
    poses: list[tuple[StereoPairObservation, np.ndarray]] = []
    K = np.asarray(camera.camera_matrix, dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera.distortion, dtype=np.float64).reshape(-1, 1)
    for pair in pairs:
        obj = np.asarray(pair.object_points, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(pair.image_points_cam1, dtype=np.float64).reshape(-1, 2)
        if obj.shape[0] < 4 or img.shape[0] < 4:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            # Rotation and translation both matter for stereo conditioning. Scale
            # translation so meters and radians live in a comparable numeric range.
            pose = np.concatenate([rvec.reshape(3), tvec.reshape(3) * 0.5])
            poses.append((pair, pose))
    if len(poses) < 2:
        return

    arr = np.asarray([pose for _pair, pose in poses], dtype=np.float64)
    center = np.mean(arr, axis=0)
    distances = np.linalg.norm(arr - center, axis=1)
    max_distance = float(max(np.max(distances), 1e-9))
    for (pair, _pose), distance in zip(poses, distances):
        score = float(max(0.0, min(100.0, distance / max_distance * 100.0)))
        pair.quality_components["pose_diversity"] = score
        if score < 15.0 and "Low pose diversity." not in pair.quality_warnings:
            pair.quality_warnings.append("Low pose diversity.")


def match_common_charuco_corners(
    detection_cam1: DetectionResult,
    detection_cam2: DetectionResult,
    *,
    pair_id: str = "pair",
    min_common_corners: int = 6,
    timestamp_cam1: float | None = None,
    timestamp_cam2: float | None = None,
    image_path_cam1: str | None = None,
    image_path_cam2: str | None = None,
    image_size_cam1: tuple[int, int] | None = None,
    image_size_cam2: tuple[int, int] | None = None,
    sync_warning_ms: float = 30.0,
) -> StereoPairObservation:
    """Match same ChArUco corner IDs between both cameras.

    Intrinsic calibration may use all detected corners per image. Stereo
    calibration must only use IDs observed by both cameras, preserving paired
    correspondence.
    """
    if not (detection_cam1.success and detection_cam2.success):
        raise ValueError("Both camera detections must be successful.")
    ids1 = _flat_ids(detection_cam1.ids)
    ids2 = _flat_ids(detection_cam2.ids)
    if ids1.size == 0 or ids2.size == 0:
        raise ValueError("Stereo ChArUco matching requires corner IDs on both detections.")

    order1 = {int(v): i for i, v in enumerate(ids1.tolist())}
    order2 = {int(v): i for i, v in enumerate(ids2.tolist())}
    common = sorted(set(order1) & set(order2))
    if not common:
        raise ValueError("No common ChArUco corner IDs between camera pair.")

    idx1 = [order1[v] for v in common]
    idx2 = [order2[v] for v in common]
    obj = np.asarray(detection_cam1.object_points, dtype=np.float32).reshape(-1, 1, 3)[idx1]
    pts1 = np.asarray(detection_cam1.corners, dtype=np.float32).reshape(-1, 1, 2)[idx1]
    pts2 = np.asarray(detection_cam2.corners, dtype=np.float32).reshape(-1, 1, 2)[idx2]

    delta_ms = None
    if timestamp_cam1 is not None and timestamp_cam2 is not None:
        delta_ms = abs(float(timestamp_cam1) - float(timestamp_cam2)) * 1000.0
    score, status, warnings = score_stereo_pair_quality(
        detection_cam1,
        detection_cam2,
        common_corners=len(common),
        min_common_corners=min_common_corners,
        sync_delta_ms=delta_ms,
        sync_warning_ms=sync_warning_ms,
    )
    components = stereo_pair_quality_components(
        detection_cam1,
        detection_cam2,
        common_corners=len(common),
        min_common_corners=min_common_corners,
        sync_delta_ms=delta_ms,
        sync_warning_ms=sync_warning_ms,
        image_size_cam1=image_size_cam1,
        image_size_cam2=image_size_cam2,
    )

    return StereoPairObservation(
        pair_id=pair_id,
        object_points=_as_point_array(obj, 3),
        image_points_cam1=_as_point_array(pts1, 2),
        image_points_cam2=_as_point_array(pts2, 2),
        common_ids=np.asarray(common, dtype=np.int32),
        detected_points_cam1=_as_point_array(detection_cam1.corners, 2),
        detected_points_cam2=_as_point_array(detection_cam2.corners, 2),
        detected_ids_cam1=_flat_ids(detection_cam1.ids),
        detected_ids_cam2=_flat_ids(detection_cam2.ids),
        timestamp_cam1=timestamp_cam1,
        timestamp_cam2=timestamp_cam2,
        image_path_cam1=image_path_cam1,
        image_path_cam2=image_path_cam2,
        sync_delta_ms=delta_ms,
        quality_score=score,
        quality_status=status,
        quality_components=components,
        quality_warnings=warnings,
    )


def build_stereo_pairs_from_datasets(
    cam1_dataset: Dataset,
    cam2_dataset: Dataset,
    *,
    min_common_corners: int = 6,
    sync_warning_ms: float = 30.0,
) -> list[StereoPairObservation]:
    pairs: list[StereoPairObservation] = []
    for index, (frame1, frame2) in enumerate(zip(cam1_dataset.frames, cam2_dataset.frames), start=1):
        if frame1.detection is None or frame2.detection is None:
            continue
        ts1 = extract_timestamp_from_sidecar(frame1.image_info.path) or extract_timestamp_from_filename(frame1.image_info.path)
        ts2 = extract_timestamp_from_sidecar(frame2.image_info.path) or extract_timestamp_from_filename(frame2.image_info.path)
        try:
            pairs.append(
                match_common_charuco_corners(
                    frame1.detection,
                    frame2.detection,
                    pair_id=f"Pair {index:03d}",
                    min_common_corners=min_common_corners,
                    timestamp_cam1=ts1,
                    timestamp_cam2=ts2,
                    image_path_cam1=frame1.image_info.path,
                    image_path_cam2=frame2.image_info.path,
                    image_size_cam1=(frame1.image_info.width, frame1.image_info.height),
                    image_size_cam2=(frame2.image_info.width, frame2.image_info.height),
                    sync_warning_ms=sync_warning_ms,
                )
            )
        except ValueError:
            continue
    apply_pair_pose_diversity_scores(pairs)
    return pairs


def set_pair_used(pair: StereoPairObservation, used: bool, reason: str | None = None) -> None:
    pair.used = bool(used)
    pair.rejected_reason = None if used else (reason or "Rejected by user")


def stats_from_values(values: list[float] | np.ndarray) -> StereoValidationStats:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return StereoValidationStats()
    return StereoValidationStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        rmse=float(np.sqrt(np.mean(arr ** 2))),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(np.max(arr)),
    )


def transformation_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inverse_rt(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_inv = np.asarray(R, dtype=np.float64).reshape(3, 3).T
    t_inv = -R_inv @ np.asarray(t, dtype=np.float64).reshape(3)
    return R_inv, t_inv.reshape(3, 1), transformation_from_rt(R_inv, t_inv)


def baseline_from_t(t: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t, dtype=np.float64).reshape(3)))


def euler_zyx_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def epipolar_errors(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    denom = np.hypot(lines2[:, 0], lines2[:, 1])
    denom[denom < 1e-12] = np.nan
    return np.abs(lines2[:, 0] * p2[:, 0] + lines2[:, 1] * p2[:, 1] + lines2[:, 2]) / denom


def sampson_distances(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((p1.shape[0], 1), dtype=np.float64)
    x1 = np.hstack([p1, ones])
    x2 = np.hstack([p2, ones])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2tFx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    denom[denom < 1e-12] = np.nan
    return (x2tFx1 ** 2) / denom


def _is_fisheye(calibration: StandardCalibration) -> bool:
    return calibration.model_name == CameraModelType.FISHEYE


def _essential_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    tx, ty, tz = np.asarray(t, dtype=np.float64).reshape(3)
    skew_t = np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)
    return skew_t @ np.asarray(R, dtype=np.float64).reshape(3, 3)


def _fundamental_from_intrinsics(E: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    return np.linalg.inv(K2).T @ np.asarray(E, dtype=np.float64).reshape(3, 3) @ np.linalg.inv(K1)


def _undistort_rectify_points(
    points: np.ndarray,
    camera: StandardCalibration,
    R: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    if _is_fisheye(camera):
        P_use = np.asarray(P, dtype=np.float64)
        if P_use.shape == (3, 4):
            P_use = P_use[:, :3]
        return cv2.fisheye.undistortPoints(
            points,
            camera.camera_matrix,
            camera.distortion.reshape(-1, 1),
            R=R,
            P=P_use,
        ).reshape(-1, 2)
    return cv2.undistortPoints(
        points,
        camera.camera_matrix,
        camera.distortion,
        R=R,
        P=P,
    ).reshape(-1, 2)


def rectification_vertical_errors(
    pairs: list[StereoPairObservation],
    camera1: StandardCalibration,
    camera2: StandardCalibration,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for pair in pairs:
        pts1 = _undistort_rectify_points(
            pair.image_points_cam1,
            camera1,
            R1,
            P1,
        )
        pts2 = _undistort_rectify_points(
            pair.image_points_cam2,
            camera2,
            R2,
            P2,
        )
        values.extend(np.abs(pts1[:, 1] - pts2[:, 1]).tolist())
    return np.asarray(values, dtype=np.float64)


def calibrate_stereo(
    pairs: list[StereoPairObservation],
    camera1: StandardCalibration,
    camera2: StandardCalibration,
    image_size: tuple[int, int],
    *,
    fix_intrinsics: bool = True,
    compute_holdout: bool = True,
    compute_audit: bool = True,
    audit_mode: str = "full",
) -> StereoCalibrationResult:
    fisheye1 = _is_fisheye(camera1)
    fisheye2 = _is_fisheye(camera2)
    if fisheye1 != fisheye2:
        raise ValueError("Mixed pinhole/fisheye stereo calibration is not supported. Use the same model for both cameras.")
    used_pairs = [p for p in pairs if p.used and p.common_count >= 4]
    if len(used_pairs) < 2:
        raise ValueError("Stereo calibration requires at least two usable pairs.")
    apply_pair_pose_diversity_scores_from_intrinsics(used_pairs, camera1)

    K1 = camera1.camera_matrix.astype(np.float64).copy()
    D1 = camera1.distortion.astype(np.float64).reshape(-1, 1).copy()
    K2 = camera2.camera_matrix.astype(np.float64).copy()
    D2 = camera2.distortion.astype(np.float64).reshape(-1, 1).copy()
    if fisheye1:
        object_points = [p.object_points.astype(np.float64).reshape(-1, 1, 3) for p in used_pairs]
        image_points1 = [p.image_points_cam1.astype(np.float64).reshape(-1, 1, 2) for p in used_pairs]
        image_points2 = [p.image_points_cam2.astype(np.float64).reshape(-1, 1, 2) for p in used_pairs]
        fix_flag = getattr(cv2.fisheye, "CALIB_FIX_INTRINSIC", cv2.CALIB_FIX_INTRINSIC)
        recompute_flag = getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", cv2.CALIB_USE_INTRINSIC_GUESS)
        flags = fix_flag if fix_intrinsics else recompute_flag
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
        rms, _K1, _D1, _K2, _D2, R, T = cv2.fisheye.stereoCalibrate(
            object_points,
            image_points1,
            image_points2,
            K1,
            D1,
            K2,
            D2,
            image_size,
            flags=flags,
            criteria=criteria,
        )[:7]
        E = _essential_from_rt(R, T)
        F = _fundamental_from_intrinsics(E, K1, K2)
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            image_size,
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
    else:
        object_points = [p.object_points.astype(np.float32) for p in used_pairs]
        image_points1 = [p.image_points_cam1.astype(np.float32) for p in used_pairs]
        image_points2 = [p.image_points_cam2.astype(np.float32) for p in used_pairs]
        flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0
        rms, _K1, _D1, _K2, _D2, R, T, E, F = cv2.stereoCalibrate(
            object_points,
            image_points1,
            image_points2,
            K1,
            D1,
            K2,
            D2,
            image_size,
            flags=flags,
        )
        R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            image_size,
            R,
            T,
        )
    T44 = transformation_from_rt(R, T)
    R_inv, t_inv, T_inv = inverse_rt(R, T)
    result = StereoCalibrationResult(
        camera1=camera1,
        camera2=camera2,
        image_size=image_size,
        stereo_rms=float(rms),
        R_cam2_from_cam1=R,
        t_cam2_from_cam1=T.reshape(3, 1),
        E=E,
        F=F,
        T_cam2_from_cam1=T44,
        R_cam1_from_cam2=R_inv,
        t_cam1_from_cam2=t_inv,
        T_cam1_from_cam2=T_inv,
        baseline=baseline_from_t(T),
        roll_pitch_yaw_deg=euler_zyx_from_rotation(R),
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        used_pair_count=len(used_pairs),
        rejected_pair_count=len([p for p in pairs if not p.used]),
        total_common_corners=sum(p.common_count for p in used_pairs),
    )
    attach_stereo_validation(result, used_pairs)
    if compute_holdout:
        attach_holdout_validation(result, used_pairs, camera1, camera2, image_size)
    if not compute_audit:
        return result
    try:
        from calibration.stereo_auditor import build_stereo_evidence_report

        evidence = build_stereo_evidence_report(
            result,
            used_pairs,
            image_size,
            full_uncertainty=audit_mode == "full",
        )
        result.capture_coach = evidence.get("capture_coach", {})
        result.sync_guard = evidence.get("sync_guard", {})
        result.calibration_audit = evidence.get("calibration_audit", {})
        result.evidence_report = evidence.get("evidence_report", {})
    except Exception as exc:  # noqa: BLE001
        result.evidence_report = {
            "confidence": "UNKNOWN",
            "warnings": [f"Evidence report generation failed: {exc}"],
        }
    return result


def attach_stereo_validation(result: StereoCalibrationResult, pairs: list[StereoPairObservation]) -> None:
    epipolar: list[float] = []
    sampson: list[float] = []
    pair_rows: list[StereoPairValidation] = []
    vertical_all = rectification_vertical_errors(
        pairs,
        result.camera1,
        result.camera2,
        result.R1,
        result.R2,
        result.P1,
        result.P2,
    )
    offset = 0
    for pair in pairs:
        epi = epipolar_errors(result.F, pair.image_points_cam1, pair.image_points_cam2)
        sam = sampson_distances(result.F, pair.image_points_cam1, pair.image_points_cam2)
        n = pair.common_count
        vert = vertical_all[offset:offset + n]
        offset += n
        epipolar.extend(epi[np.isfinite(epi)].tolist())
        sampson.extend(sam[np.isfinite(sam)].tolist())
        epi_mean = float(np.mean(epi)) if epi.size else None
        vert_mean = float(np.mean(vert)) if vert.size else None
        pair_rows.append(StereoPairValidation(
            pair_id=pair.pair_id,
            common_corners=pair.common_count,
            epipolar_mean=epi_mean,
            sampson_mean=float(np.mean(sam)) if sam.size else None,
            vertical_mean=vert_mean,
        ))
    result.epipolar_error = stats_from_values(epipolar)
    result.sampson_distance = stats_from_values(sampson)
    result.rectification_vertical_error = stats_from_values(vertical_all)
    result.pair_validations = pair_rows
    mark_stereo_outlier_candidates(result)


def _split_pairs(
    pairs: list[StereoPairObservation],
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[StereoPairObservation], list[StereoPairObservation]]:
    if len(pairs) < 5:
        return pairs, []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(pairs) * test_ratio)))
    test_idx = set(indices[:n_test].tolist())
    train = [p for i, p in enumerate(pairs) if i not in test_idx]
    test = [p for i, p in enumerate(pairs) if i in test_idx]
    return train, test


def _validation_epipolar_stats(result: StereoCalibrationResult, pairs: list[StereoPairObservation]) -> StereoValidationStats:
    values: list[float] = []
    for pair in pairs:
        errors = epipolar_errors(result.F, pair.image_points_cam1, pair.image_points_cam2)
        values.extend(errors[np.isfinite(errors)].tolist())
    return stats_from_values(values)


def attach_holdout_validation(
    result: StereoCalibrationResult,
    pairs: list[StereoPairObservation],
    camera1: StandardCalibration,
    camera2: StandardCalibration,
    image_size: tuple[int, int],
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    train_pairs, validation_pairs = _split_pairs(pairs, test_ratio=test_ratio, seed=seed)
    result.holdout_train_pair_count = len(train_pairs)
    result.holdout_validation_pair_count = len(validation_pairs)
    if not validation_pairs or len(train_pairs) < 2:
        return
    train_result = calibrate_stereo(
        train_pairs,
        camera1,
        camera2,
        image_size,
        fix_intrinsics=True,
        compute_holdout=False,
    )
    result.holdout_training_error = _validation_epipolar_stats(train_result, train_pairs)
    result.holdout_validation_error = _validation_epipolar_stats(train_result, validation_pairs)
    if result.holdout_training_error.rmse is not None and result.holdout_validation_error.rmse is not None:
        result.holdout_generalization_gap = (
            result.holdout_validation_error.rmse - result.holdout_training_error.rmse
        )
