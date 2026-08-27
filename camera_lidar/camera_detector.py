"""
camera_calibrator.camera_lidar.camera_detector
==================================================

Camera-side FAST-Calib target detection: ArUco corner markers -> board pose
(PnP) -> 4 circle-center 3D positions in the camera frame, in
target_config.CORNER_ORDER (a real-world semantic order, since it is
derived from the known marker-ID -> corner assignment in TargetConfig).

Independent reimplementation (see camera_lidar/types.py module docstring
for license/provenance notes) using OpenCV's cv2.aruco module and
cv2.solvePnP over the combined marker corners -- not a port of upstream's
estimatePoseBoard/projectPoints C++ code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from calibration.calibration_io import StandardCalibration
from camera_lidar.target_config import TargetConfig
from camera_lidar.types import FailureReason

_MIN_MARKERS_FOR_PNP = 3


@dataclass
class CameraDetectionResult:
    success: bool
    failure_reason: Optional[FailureReason] = None
    circle_centers: Optional[np.ndarray] = None    # (4,3) camera frame, CORNER_ORDER
    # circle_centers above is ALWAYS all 4 (pose-inferred from the fitted
    # board rigid-body transform, once >=3 markers give a valid PnP solve --
    # geometry, not a guess). detected_ids is the SUBSET whose OWN marker was
    # actually, independently seen -- correspondence/classification code
    # must only trust circle_centers entries whose canonical id is in
    # detected_ids (never treat a pose-inferred-only corner as "detected",
    # per the "don't insert an estimated missing center" requirement).
    detected_ids: frozenset = field(default_factory=frozenset)
    markers_detected: int = 0
    markers_expected: int = 4
    reprojection_error_px: Optional[float] = None
    board_pose_rvec: Optional[np.ndarray] = None
    board_pose_tvec: Optional[np.ndarray] = None
    detected_marker_ids: list[int] = field(default_factory=list)
    detected_corners_image: Optional[list[np.ndarray]] = None  # for overlay/debug rendering

    # Diagnostic fields (Marker Extraction Diagnostic Mode) -- populated on
    # EVERY return path below, including both failure branches, since "what
    # did the raw detector actually see" is exactly what a diagnostic view
    # needs when the pipeline as a whole failed.
    rejected_candidate_count: int = 0     # marker-like quadrilaterals the detector saw but could not decode
    dictionary_name: str = ""             # target.aruco_dictionary, carried alongside the result for self-contained display
    matched_marker_ids: list[int] = field(default_factory=list)   # raw ids that matched an expected id (subset of detected_marker_ids)
    missing_marker_ids: list[int] = field(default_factory=list)   # expected ids with no raw match, in raw-id space (not corner names)


def _aruco_dictionary(name: str):
    dict_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)  # pragma: no cover -- older opencv-contrib fallback


def _detect_markers(gray: np.ndarray, aruco_dict) -> tuple[list, Optional[np.ndarray], list]:
    """Returns (corners, ids, rejected_candidates). rejected_candidates are
    marker-like quadrilaterals the detector found but could not decode into
    a valid id -- an important diagnostic signal (see
    CameraDetectionResult.rejected_candidate_count): "0 detected, 0
    rejected" (nothing quad-like found at all) and "0 detected, 30 rejected"
    (quads found but decoding failed -- wrong dictionary / damaged border /
    low resolution) point at very different root causes."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:  # pragma: no cover -- older opencv-contrib fallback
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected


def detect_camera_target(
    image: np.ndarray,
    intrinsics: StandardCalibration,
    target: TargetConfig,
) -> CameraDetectionResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    aruco_dict = _aruco_dictionary(target.aruco_dictionary)
    corners, ids, rejected = _detect_markers(gray, aruco_dict)
    rejected_candidate_count = len(rejected) if rejected is not None else 0

    marker_centers_board = target.marker_centers_board_frame()
    expected_ids = set(marker_centers_board.keys())

    if ids is None or len(ids) == 0:
        return CameraDetectionResult(
            success=False,
            failure_reason=FailureReason.CAMERA_MARKER_NOT_FOUND,
            markers_expected=len(expected_ids),
            dictionary_name=target.aruco_dictionary,
            rejected_candidate_count=rejected_candidate_count,
            missing_marker_ids=sorted(expected_ids),
        )

    ids_flat = ids.flatten().tolist()
    found_ids = [i for i in ids_flat if i in expected_ids]
    missing_marker_ids = sorted(expected_ids - set(found_ids))
    corner_by_marker_id = {marker_id: corner for corner, marker_id in target.marker_ids.items()}
    detected_ids = frozenset(corner_by_marker_id[i] for i in found_ids if i in corner_by_marker_id)

    if len(found_ids) < _MIN_MARKERS_FOR_PNP:
        # A stable planar PnP solve needs at least 3 non-collinear markers.
        return CameraDetectionResult(
            success=False,
            failure_reason=FailureReason.CAMERA_MARKER_NOT_FOUND,
            markers_detected=len(found_ids),
            markers_expected=len(expected_ids),
            detected_marker_ids=ids_flat,
            detected_ids=detected_ids,
            detected_corners_image=list(corners),
            dictionary_name=target.aruco_dictionary,
            rejected_candidate_count=rejected_candidate_count,
            matched_marker_ids=found_ids,
            missing_marker_ids=missing_marker_ids,
        )

    # Build 2D<->3D correspondences: each recognized marker contributes its 4
    # image corners and the matching 4 board-frame 3D corners (a marker_size
    # square centered on that marker's known board-frame position).
    half = target.marker_size / 2.0
    # cv2.aruco corner order is (top-left, top-right, bottom-right,
    # bottom-left) of the marker as seen in the image; mirror that ordering
    # in the board-frame square around the marker center.
    local_corner_offsets = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ])

    object_points = []
    image_points = []
    for marker_id, corner_pts in zip(ids_flat, corners):
        if marker_id not in marker_centers_board:
            continue
        center = marker_centers_board[marker_id]
        object_points.append(center + local_corner_offsets)
        image_points.append(corner_pts.reshape(4, 2))

    object_points = np.concatenate(object_points, axis=0).astype(np.float64)
    image_points = np.concatenate(image_points, axis=0).astype(np.float64)

    camera_matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1, 1)

    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return CameraDetectionResult(
            success=False,
            failure_reason=FailureReason.CAMERA_MARKER_NOT_FOUND,
            markers_detected=len(found_ids),
            markers_expected=len(expected_ids),
            detected_marker_ids=ids_flat,
            detected_ids=detected_ids,
            detected_corners_image=list(corners),
            dictionary_name=target.aruco_dictionary,
            rejected_candidate_count=rejected_candidate_count,
            matched_marker_ids=found_ids,
            missing_marker_ids=missing_marker_ids,
        )

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
    )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    board_circle_centers = target.circle_centers_board_frame()   # (4,3), CORNER_ORDER
    camera_circle_centers = (R @ board_circle_centers.T).T + t

    return CameraDetectionResult(
        success=True,
        circle_centers=camera_circle_centers,
        detected_ids=detected_ids,
        markers_detected=len(found_ids),
        markers_expected=len(expected_ids),
        reprojection_error_px=reprojection_error,
        board_pose_rvec=rvec,
        board_pose_tvec=tvec,
        detected_marker_ids=ids_flat,
        detected_corners_image=list(corners),
        dictionary_name=target.aruco_dictionary,
        rejected_candidate_count=rejected_candidate_count,
        matched_marker_ids=found_ids,
        missing_marker_ids=missing_marker_ids,
    )


# Public (not `_`-prefixed): also used by ui/camera_lidar_workspace.py to
# populate the Target Geometry dictionary combo box, so the diagnostic
# candidate list and the configurable-dictionary list never drift apart.
COMMON_ARUCO_DICTIONARIES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
    "DICT_5X5_50", "DICT_5X5_100",
    "DICT_6X6_50", "DICT_6X6_250",
    "DICT_7X7_50",
]


def render_marker_overlay(image: np.ndarray, result: CameraDetectionResult) -> np.ndarray:
    """Draws ALL raw detected marker corners + ids (not just the expected-id
    subset) via cv2.aruco.drawDetectedMarkers -- a diagnostic overlay that
    works independently of whether detect_camera_target's PnP solve
    succeeded, since result.detected_corners_image/detected_marker_ids are
    now populated on every return path (see CameraDetectionResult)."""
    overlay = image.copy()
    if result.detected_corners_image and result.detected_marker_ids:
        ids_array = np.array(result.detected_marker_ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(overlay, result.detected_corners_image, ids_array)
    return overlay


def diagnose_dictionaries(
    image: np.ndarray, candidate_dictionaries: Optional[list[str]] = None,
) -> list[tuple[str, int]]:
    """Diagnostic-only: tries each candidate ArUco dictionary against
    `image` (reusing _detect_markers -- no separate detection logic) and
    reports (dictionary_name, raw_marker_count) sorted by count descending.
    Never mutates a TargetConfig -- purely informational; the caller decides
    whether to act on the recommendation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    candidates = candidate_dictionaries or COMMON_ARUCO_DICTIONARIES
    results = []
    for name in candidates:
        try:
            aruco_dict = _aruco_dictionary(name)
        except AttributeError:
            continue
        _corners, ids, _rejected = _detect_markers(gray, aruco_dict)
        count = 0 if ids is None else len(ids)
        results.append((name, count))
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results
