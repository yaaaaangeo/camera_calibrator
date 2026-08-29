"""
camera_calibrator.camera_lidar.types
=======================================

Common Data Model for the Camera-LiDAR FAST-Calib feature, plus the
result/failure-reason types shared by the whole pipeline.

Provenance / license note: the target-detection and calibration algorithm
implemented across camera_lidar/ is an independent Python reimplementation
of the pipeline described in the paper "FAST-Calib: LiDAR-Camera Extrinsic
Calibration in One Second" and the parameter *schema* of upstream
hku-mars/FAST-Calib's config/qr_params.yaml (which parameters exist, e.g.
circle_radius / delta_width_circles). It is not a port of upstream's
GPL-2.0 C++ source (src/*.hpp, src/*.cpp) -- this project is MIT licensed,
so the math here (RANSAC plane fit, boundary/circle extraction, SVD rigid-
transform solve) is written from the published pipeline description and
standard point-set-registration references, not translated from the GPL
sources.

Dependency direction (enforced by construction, not by convention):

    Adapter (bag / live / file) -> Common Data Model -> camera_lidar core

Nothing in camera_lidar/ imports rospy, rclpy, rosbags,
calibration.ros_live, or calibration.rosbag_reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from calibration.calibration_io import StandardCalibration
from camera_lidar.target_config import TargetConfig


@dataclass
class ImageFrame:
    timestamp: float
    image: np.ndarray                       # BGR, HxWx3 (OpenCV convention)
    frame_id: str = ""
    source_metadata: dict = field(default_factory=dict)


@dataclass
class PointCloudFrame:
    timestamp: float
    points: np.ndarray                      # (N, 3) float
    frame_id: str = ""
    intensity: Optional[np.ndarray] = None  # (N,)
    ring: Optional[np.ndarray] = None       # (N,)
    source_metadata: dict = field(default_factory=dict)


@dataclass
class ROIConfig:
    """Distance/ROI filter applied to the point cloud before plane
    segmentation. Mirrors upstream FAST-Calib's distance-filter concept."""
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = -10.0
    z_max: float = 10.0


@dataclass
class CalibrationScene:
    image: ImageFrame
    cloud: PointCloudFrame
    intrinsics: StandardCalibration
    target: TargetConfig
    roi: ROIConfig = field(default_factory=ROIConfig)
    metadata: dict = field(default_factory=dict)


class FailureReason(Enum):
    CANCELLED = "cancelled"
    CAMERA_MARKER_NOT_FOUND = "camera_marker_not_found"
    LIDAR_PLANE_NOT_FOUND = "lidar_plane_not_found"
    INSUFFICIENT_ROI_POINTS = "insufficient_roi_points"
    CIRCLES_NOT_FOUND = "circles_not_found"
    CIRCLE_FIT_UNSTABLE = "circle_fit_unstable"
    GEOMETRY_MISMATCH = "geometry_mismatch"
    NOT_ENOUGH_VALID_SCENES = "not_enough_valid_scenes"
    INSUFFICIENT_COMMON_FEATURES = "insufficient_common_features"


_FAILURE_MESSAGES: dict[FailureReason, str] = {
    FailureReason.CANCELLED: "Calibration cancelled by user.",
    FailureReason.CAMERA_MARKER_NOT_FOUND: (
        "Camera marker detection failed: could not detect enough ArUco markers "
        "on the target board. Ensure the board is fully visible, in focus, and "
        "well lit."
    ),
    FailureReason.LIDAR_PLANE_NOT_FOUND: (
        "LiDAR plane not found: RANSAC plane segmentation did not find a "
        "dominant planar surface inside the ROI. Check the ROI bounds and "
        "that the board is inside them."
    ),
    FailureReason.INSUFFICIENT_ROI_POINTS: (
        "Insufficient ROI points: too few LiDAR points fell inside the "
        "configured ROI. Widen the ROI or move the board closer to the sensor."
    ),
    FailureReason.CIRCLES_NOT_FOUND: (
        "Expected circles not found: fewer than 4 circle candidates were "
        "extracted from the plane boundary. Check the target is not occluded "
        "and circle_radius matches the physical board."
    ),
    FailureReason.CIRCLE_FIT_UNSTABLE: (
        "Circle fitting unstable: circle fits did not converge to consistent "
        "radii/centers. Try a denser point cloud (closer range or accumulate "
        "more scans)."
    ),
    FailureReason.GEOMETRY_MISMATCH: (
        "Target geometry mismatch: detected circle centers do not match the "
        "configured board geometry (delta_width/height_circles). Check "
        "TargetConfig matches the physical board."
    ),
    FailureReason.NOT_ENOUGH_VALID_SCENES: (
        "Not enough valid scenes: fewer than 2 captured scenes produced a "
        "successful single-scene detection. Multi-scene calibration needs at "
        "least 2 valid scenes -- capture more, or check why existing scenes "
        "are failing detection in the Scene Manager."
    ),
    FailureReason.INSUFFICIENT_COMMON_FEATURES: (
        "Insufficient common features: camera and LiDAR did not agree on at "
        "least 3 of the same canonical corners (top_left/top_right/bottom_left/"
        "bottom_right). A feature only counts if BOTH sensors independently "
        "detected it -- check the per-sensor diagnostics to see which corner(s) "
        "each side is missing."
    ),
}


def failure_message(reason: FailureReason) -> str:
    return _FAILURE_MESSAGES[reason]


class SceneType(Enum):
    """Scene classification by COMMON feature count (camera_ids ∩ lidar_ids,
    not either sensor's raw count alone) -- see camera_lidar.pipeline for
    where this gets computed. A real enum per the requirement to never
    compare scene classification via bare strings."""
    VALID_FULL = "valid_full"        # 4/4 common features
    VALID_PARTIAL = "valid_partial"  # 3/4 common features, gated (quality/geometry/stability)
    INVALID = "invalid"              # <=2/4 common features


@dataclass
class CameraLidarCalibrationResult:
    success: bool
    failure_reason: Optional[FailureReason] = None

    R_camera_from_lidar: Optional[np.ndarray] = None   # 3x3
    t_camera_from_lidar: Optional[np.ndarray] = None   # (3,)
    T_camera_from_lidar: Optional[np.ndarray] = None   # 4x4
    T_lidar_from_camera: Optional[np.ndarray] = None   # 4x4

    camera_centers: Optional[np.ndarray] = None  # (4,3), camera frame, matched order
    lidar_centers: Optional[np.ndarray] = None   # (4,3), lidar frame, matched order

    residual_rmse_m: Optional[float] = None
    residual_mean_m: Optional[float] = None
    residual_median_m: Optional[float] = None
    residual_p95_m: Optional[float] = None
    residual_max_m: Optional[float] = None

    # Raw per-stage detection results (CameraDetectionResult / LidarDetectionResult),
    # attached by camera_lidar.pipeline.calibrate_single_scene purely so the UI's
    # diagnostics panel can show ROI/plane/boundary/circle-fit detail without
    # re-running detection. Untyped here (Any) to avoid a circular import --
    # camera_detector.py/lidar_detector.py both import FailureReason from this
    # module, so this module cannot import their result types back.
    camera_detection: Optional[Any] = None
    lidar_detection: Optional[Any] = None

    # Canonical common-feature classification (camera_lidar.pipeline).
    scene_type: Optional[SceneType] = None
    common_ids: frozenset = field(default_factory=frozenset)    # canonical corner names in BOTH sensors
    missing_from_camera: frozenset = field(default_factory=frozenset)
    missing_from_lidar: frozenset = field(default_factory=frozenset)
    # True when a PARTIAL (3/4) correspondence had multiple congruent
    # corner-hypotheses tie and no reference_transform was available to
    # break the tie -- see camera_lidar/correspondence.py's module
    # docstring. The Quality Gate treats this as an automatic fail.
    correspondence_ambiguous: bool = False

    @property
    def error_message(self) -> Optional[str]:
        if self.failure_reason is None:
            return None
        return failure_message(self.failure_reason)


@dataclass
class CapturedScene:
    """One captured Scene in the Scene Manager: a CalibrationScene plus
    bookkeeping the manager/UI needs (whether it's included in the next
    Multi-Scene solve, which topics it came from -- for the sensor-pair-
    change warning -- and a stable id for the scene table)."""
    scene_id: str
    scene: CalibrationScene
    included: bool = True
    camera_topic: str = ""
    lidar_topic: str = ""
    roi_mode: str = "auto"  # "auto" | "manual"
    detection: Optional[CameraLidarCalibrationResult] = None  # single-scene provisional result, for the table


@dataclass
class SceneResidual:
    """How well the FINAL multi-scene T_camera_from_lidar fits one
    scene's own 4 correspondences (re-applying the joint transform to that
    scene's points -- distinct from that scene's own single-scene
    provisional fit)."""
    scene_id: str
    rmse_m: float
    p95_m: float
    is_outlier: bool


@dataclass
class MultiSceneResult(CameraLidarCalibrationResult):
    """Final multi-scene calibration result. Extends
    CameraLidarCalibrationResult (same R/t/T/residual fields, computed over
    ALL included scenes' pooled correspondences jointly) with per-scene
    breakdown and outlier flags."""
    scene_count: int = 0
    per_scene: list[SceneResidual] = field(default_factory=list)
    outlier_scene_ids: list[str] = field(default_factory=list)
    policy: str = "strict"  # "strict" | "flexible" -- which scenes were pooled


@dataclass
class TargetPose:
    """Shared pose summary for the Stability and Duplicate-Pose gates,
    derived from a scene's matched circle centers (camera OR lidar frame --
    callers should be consistent about which)."""
    position: np.ndarray       # (3,) centroid of the matched centers
    plane_normal: np.ndarray   # (3,) unit normal of the best-fit plane through the centers
    distance: float            # ||position|| -- range from sensor origin


@dataclass
class DegenerateGeometryResult:
    """3-point PARTIAL-scene geometry sanity check (camera_lidar.gates) --
    3 correspondences CAN solve a rigid transform, but not reliably if the
    3 points are nearly collinear."""
    passed: bool
    triangle_area_m2: float
    min_pairwise_distance_m: float
    reason: Optional[str] = None


@dataclass
class QualityGateResult:
    passed: bool
    reason: Optional[str] = None
    metrics: dict = field(default_factory=dict)


@dataclass
class StabilityGateResult:
    passed: bool
    reason: Optional[str] = None
    position_change_m: Optional[float] = None
    normal_change_deg: Optional[float] = None


@dataclass
class DuplicateGateResult:
    passed: bool
    reason: Optional[str] = None
    nearest_scene_id: Optional[str] = None
    position_difference_m: Optional[float] = None
    orientation_difference_deg: Optional[float] = None


@dataclass
class SceneCandidate:
    """One auto-discovered candidate from camera_lidar.scene_extraction's
    bag-wide scan: one representative frame from a "Stable Scene Segment"
    (a run of consecutive frames with the same detected marker IDs and a
    settled target pose), paired with the nearest LiDAR cloud by timestamp.

    Classification here is CAMERA-ONLY and preliminary (based on how many of
    the target's *expected* ArUco markers this frame saw) -- it is NOT the
    same as CameraLidarCalibrationResult.scene_type, which is the real
    camera-LiDAR common-feature classification computed once a candidate is
    actually run through calibrate_single_scene (Scene Manager "ADD SELECTED").

    `is_selected` is deliberately the only mutable field the UI touches for
    selection state, and it lives on this object (not the table widget) so
    that switching the Scene Browser's display filter (ALL / 4 MARKERS /
    3 MARKERS) can never affect which candidates are selected."""
    candidate_id: str
    segment_start_s: float
    segment_end_s: float
    representative_timestamp_s: float
    camera_topic: str
    lidar_topic: str
    image: np.ndarray
    camera_detection: Any  # CameraDetectionResult -- Any to avoid a circular import, see CameraLidarCalibrationResult above
    scene_type: SceneType  # camera-only preliminary FULL/PARTIAL classification (INVALID candidates are never built)
    detected_ids: frozenset = field(default_factory=frozenset)
    missing_ids: frozenset = field(default_factory=frozenset)
    quality_score: float = 0.0
    cloud_points: Optional[np.ndarray] = None
    cloud_timestamp_s: Optional[float] = None
    is_selected: bool = False


@dataclass
class PolicyComparisonResult:
    """STRICT vs FLEXIBLE Multi-Scene result comparison (§29-31)."""
    strict_result: MultiSceneResult
    flexible_result: MultiSceneResult
    translation_difference_m: Optional[float] = None
    rotation_difference_deg: Optional[float] = None
    residual_difference_m: Optional[float] = None
    impact: Optional[str] = None  # "LOW" | "HIGH" | None (None if either side failed)
