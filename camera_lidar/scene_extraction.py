"""
camera_calibrator.camera_lidar.scene_extraction
====================================================

Bag-wide "MARKER EXTRACTION" scan: run camera-side ArUco detection on every
frame of a camera topic, group consecutive frames with an unchanged marker
set and a settled target pose into "Stable Scene Segments", and pick one
best-quality representative frame per segment. Each representative becomes a
SceneCandidate the Scene Browser UI can show and let the user check/uncheck.

Design principle (from this feature's spec): camera ArUco discovery and
LiDAR processing are two separate stages. This module only ever runs the
cheap camera-side detector (camera_detector.detect_camera_target, reused
unchanged -- no new detection algorithm) across the whole scan; the
expensive LiDAR AUTO-ROI multi-plane search only runs later, once per scene
the user actually selects (camera_lidar.pipeline.calibrate_single_scene via
the Scene Manager's existing "ADD SELECTED" flow), not once per scanned
frame.

ROS-independence: like the rest of camera_lidar/, this module never imports
rosbags or calibration.rosbag_reader (see camera_lidar/types.py's
dependency-direction note). build_scene_candidates() takes a `frames_factory`
(a zero-arg callable returning a fresh (image, timestamp_s, frame_id)
iterator each time it's called -- e.g. a bag reader generator function) and a
`cloud_lookup` closure instead of a bag path, so the adapter (bag today, Live
later) supplies the actual I/O. calibration.camera_lidar_controller wires
this module to calibration.rosbag_reader for the Bag pathway.
"""

from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

import numpy as np

from calibration.calibration_io import StandardCalibration
from camera_lidar.camera_detector import CameraDetectionResult, detect_camera_target
from camera_lidar.extraction_diagnostics import ExtractionDiagnosticSummary, ExtractionDiagnosticTracker
from camera_lidar.gates import StabilityThresholds, compute_target_pose, evaluate_stability_gate
from camera_lidar.target_config import CORNER_ORDER, TargetConfig
from camera_lidar.types import SceneCandidate, SceneType


def classify_candidate(camera_result: CameraDetectionResult, target: TargetConfig) -> SceneType:
    """Preliminary, CAMERA-ONLY classification by count of the target's
    *expected* marker IDs actually, independently detected (detected_ids --
    never a raw ArUco marker count, which could include stray background
    markers not on the board at all)."""
    expected = len(target.marker_ids)
    count = len(camera_result.detected_ids)
    if count >= expected:
        return SceneType.VALID_FULL
    if count == expected - 1:
        return SceneType.VALID_PARTIAL
    return SceneType.INVALID


def _frame_quality_score(camera_result: CameraDetectionResult) -> float:
    """More markers first, then lower reprojection error -- used both to
    pick each segment's representative frame and to display in the Scene
    Browser."""
    reprojection_error = camera_result.reprojection_error_px or 0.0
    return camera_result.markers_detected * 100.0 - reprojection_error


_SEGMENT_LOOKBACK_FRAMES = 5


class _SegmentBoundaryTracker:
    """The per-frame "does this frame continue the current Stable Scene
    Segment, or start a new one" decision, factored out so
    detect_stable_segments (batch, index-only, used by tests/anyone
    reasoning about segmentation in isolation) and build_scene_candidates
    (single streaming pass, image-aware) share EXACTLY the same logic
    instead of two copies that could silently drift apart.

    A segment tracks a `reference_ids` set -- the union of every
    detected_ids set seen so far in the segment -- and a frame continues the
    segment if its own ids are a SUBSET or SUPERSET of that reference (i.e.
    a marker that's momentarily missing, or one that reappears, is still
    "the same hold"; only a DIFFERENT id showing up that the reference has
    never seen means the target's marker visibility genuinely changed) AND
    its pose stays within gates.evaluate_stability_gate's thresholds of the
    last `_SEGMENT_LOOKBACK_FRAMES` poses in the segment (not strictly the
    immediately-previous frame).

    Real bag footage has per-frame detection noise synthetic test data
    doesn't: a marker can flicker out for a single frame (motion blur/
    lighting) and PnP pose solves jitter a little even when the physical
    target is dead still. A strict single-previous-frame/exact-ids
    comparison treats every such blip as "the target moved" and starts a
    brand new segment -- on real noisy footage this produced 1000+ segments
    from what should have been a handful of held poses (a real user-reported
    bug). The pose-window check remains the real gatekeeper for "is this
    actually the same physical hold" -- ids alone can't reliably tell a
    brief dropout from an actual move, since with only 4 possible canonical
    corner names, any partial (3-id) reading is trivially a subset of a full
    (4-id) reference regardless of cause; the ids check still catches a
    genuinely different partial pattern (e.g. one 3-id set replaced by a
    disjoint-ish different 3-id set) that the pose check alone might not.
    """

    def __init__(self, thresholds: Optional[StabilityThresholds] = None):
        self.thresholds = thresholds or StabilityThresholds()
        self.reference_ids: Optional[frozenset] = None
        self.recent_poses: list = []

    def reset(self) -> None:
        self.reference_ids = None
        self.recent_poses = []

    def observe(self, ids: frozenset, pose) -> bool:
        """Returns True if this frame starts a NEW segment (the caller is
        responsible for finalizing whatever was open before this call).
        Always updates internal state to include this frame either way --
        on a "starts new" result the tracker has already reset and re-seeded
        itself with THIS frame, so the caller must not call reset() again
        afterward (that would wipe the very state this frame just seeded)."""
        ids_compatible = self.reference_ids is None or ids <= self.reference_ids or ids >= self.reference_ids
        pose_compatible = not self.recent_poses or evaluate_stability_gate(pose, self.recent_poses, self.thresholds).passed
        starts_new = not (ids_compatible and pose_compatible)
        if starts_new:
            self.reset()
        self.reference_ids = ids if self.reference_ids is None else (self.reference_ids | ids)
        self.recent_poses.append(pose)
        if len(self.recent_poses) > _SEGMENT_LOOKBACK_FRAMES:
            self.recent_poses.pop(0)
        return starts_new


def _stability_pose(camera_result: CameraDetectionResult):
    # Use ALL 4 pose-inferred circle_centers (not just the detected_ids
    # subset) for stability tracking: circle_centers is the board's rigid-
    # body PnP fit and stays consistent across frames regardless of exactly
    # which >=3 markers independently confirmed it this frame, whereas a
    # centroid over only the currently-visible subset shifts by centimeters
    # (board-geometry-dependent) the moment one marker flickers out -- that
    # shift alone would break a segment on "pose moved" grounds even though
    # the physical target never moved, defeating the ids-subset tolerance
    # above. detected_ids-filtering still matters for correspondence/
    # classification (never treat a pose-inferred-only corner as
    # independently confirmed there) but not for this stability comparison.
    return compute_target_pose(camera_result.circle_centers)


def detect_stable_segments(
    observations: list[tuple[float, CameraDetectionResult]],
    thresholds: Optional[StabilityThresholds] = None,
) -> list[list[int]]:
    """Batch/index-only Stable Scene Segment grouping over an already-
    collected `observations` list -- see _SegmentBoundaryTracker for the
    actual per-frame decision logic this shares with build_scene_candidates'
    streaming pass. A detection failure always starts a new segment.
    Returns lists of indices into `observations`."""
    tracker = _SegmentBoundaryTracker(thresholds)
    segments: list[list[int]] = []
    current: list[int] = []

    for idx, (_timestamp, camera_result) in enumerate(observations):
        if not camera_result.success or len(camera_result.detected_ids) < 3:
            if current:
                segments.append(current)
            current = []
            tracker.reset()
            continue

        starts_new = tracker.observe(camera_result.detected_ids, _stability_pose(camera_result))
        if starts_new and current:
            segments.append(current)
            current = []
        current.append(idx)

    if current:
        segments.append(current)
    return segments


def select_representative(segment_observations: list[tuple[float, CameraDetectionResult]]) -> int:
    """Index (within `segment_observations`) of the highest-quality frame in
    a segment -- argmax of _frame_quality_score, not the first frame."""
    scores = [_frame_quality_score(camera_result) for _timestamp, camera_result in segment_observations]
    return int(np.argmax(scores))


def build_scene_candidates(
    frames_factory: Callable[[], Iterable[tuple[np.ndarray, float, str]]],
    camera_topic: str,
    lidar_topic: str,
    intrinsics: StandardCalibration,
    target: TargetConfig,
    cloud_lookup: Callable[[float], Optional[tuple[np.ndarray, float]]],
    thresholds: Optional[StabilityThresholds] = None,
    total_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    pair_lidar: bool = True,
    detector_workers: int = 1,
) -> tuple[list[SceneCandidate], ExtractionDiagnosticSummary]:
    """Single streaming pass over `frames_factory()`: detects the camera
    target on each frame, tracks Stable Scene Segment boundaries online
    (_SegmentBoundaryTracker -- the exact same logic detect_stable_segments
    uses batch), and keeps only the CURRENT open segment's single best-
    quality frame IMAGE in memory at a time, finalizing it into a
    SceneCandidate (FULL/PARTIAL only -- INVALID segments are dropped, they
    were never real candidates) as soon as the segment closes.

    This replaced an earlier two-pass design (a full detection pass, then a
    full SECOND decode-only pass just to fetch representative images) that
    turned out to be a serious problem in practice: for a bag that
    fragments into many hundreds of segments, representative frames end up
    scattered across virtually the whole recording, so the "second pass"
    was effectively a second full bag scan -- for a large/slow bag, users
    reported it appearing to hang for 30+ minutes with no way to tell it
    apart from a real deadlock. Streaming calls `frames_factory()` exactly
    ONCE, so that failure mode can no longer happen, and total scan time is
    roughly halved (no redundant second decode of the whole bag).

    `total_frames` (if known -- e.g. the adapter can read it from the bag's
    own topic metadata) only affects progress reporting: it lets
    `frame_progress_callback` report real percentage instead of an
    unbounded count, mirroring calibration.rosbag_reader.extract_images_from_bag's
    (done, total) progress convention. Detection itself does not need it.

    Returns (candidates, summary) -- summary is an ExtractionDiagnosticSummary
    (camera_lidar.extraction_diagnostics) built alongside the scan via one
    `tracker.observe(...)` call per frame, so "0 candidates" always comes
    with a stage-by-stage funnel explaining where the data was lost, not a
    bare empty list.

    If `pair_lidar` is False, candidate creation intentionally skips
    `cloud_lookup`; callers can load the nearest LiDAR frame later only for
    the candidates the user actually selects. This keeps camera-only marker
    extraction fast on bags that produce many candidate segments.

    `detector_workers` can parallelize per-frame ArUco detection while
    preserving ordered segmentation. Every frame still uses the same
    full-resolution detector; only scheduling changes.

    `cancel_check` (checked every frame): a scan over a large bag can take a
    long time with no way to know in advance how long -- without this there
    was no way to stop one short of killing the whole app (a real user-
    reported situation). The segment open at the moment of cancellation is
    still finalized (nothing already found is thrown away, matching
    calibration.rosbag_reader.extract_images_from_bag's existing cancel
    convention).
    """
    thresholds = thresholds or StabilityThresholds()
    tracker = ExtractionDiagnosticTracker()
    tracker.summary.expected_marker_ids = sorted(target.marker_ids.values())

    boundary = _SegmentBoundaryTracker(thresholds)
    candidates: list[SceneCandidate] = []
    segment_count = 0
    segment_start_s: Optional[float] = None
    segment_end_s: Optional[float] = None
    best_in_segment = None  # (score, representative_timestamp_s, camera_result, image)

    def _finalize_open_segment() -> None:
        nonlocal segment_count, segment_start_s, segment_end_s, best_in_segment
        if best_in_segment is not None:
            segment_count += 1
            _score, rep_timestamp_s, camera_result, image = best_in_segment
            scene_type = classify_candidate(camera_result, target)
            if scene_type != SceneType.INVALID:
                cloud_points, cloud_timestamp_s = None, None
                if pair_lidar:
                    looked_up = cloud_lookup(rep_timestamp_s)
                    if looked_up is not None:
                        cloud_points, cloud_timestamp_s = looked_up
                candidates.append(SceneCandidate(
                    candidate_id=f"candidate_{len(candidates) + 1:03d}",
                    segment_start_s=segment_start_s,
                    segment_end_s=segment_end_s,
                    representative_timestamp_s=rep_timestamp_s,
                    camera_topic=camera_topic,
                    lidar_topic=lidar_topic,
                    image=image,
                    camera_detection=camera_result,
                    scene_type=scene_type,
                    detected_ids=camera_result.detected_ids,
                    missing_ids=frozenset(CORNER_ORDER) - camera_result.detected_ids,
                    quality_score=_score,
                    cloud_points=cloud_points,
                    cloud_timestamp_s=cloud_timestamp_s,
                ))
        segment_start_s, segment_end_s, best_in_segment = None, None, None

    processed = 0
    start_monotonic = time.monotonic()

    def _process_detection(image: np.ndarray, timestamp_s: float, camera_result: CameraDetectionResult) -> None:
        nonlocal processed, segment_start_s, segment_end_s, best_in_segment
        tracker.observe(camera_result)
        processed += 1
        if frame_progress_callback is not None:
            frame_progress_callback(processed, total_frames or 0)
        if progress_callback is not None and processed % 20 == 0:
            elapsed_s = time.monotonic() - start_monotonic
            rate = processed / elapsed_s if elapsed_s > 0 else 0.0
            elapsed_text = f"elapsed {elapsed_s:.1f}s, {rate:.1f} fps"
            if total_frames:
                pct = processed / total_frames * 100.0
                progress_callback(
                    f"Scanning camera topic... {processed}/{total_frames} frames ({pct:.0f}%), "
                    f"{len(candidates)} candidate(s) so far, {elapsed_text}"
                )
            else:
                progress_callback(
                    f"Scanning camera topic... {processed} frames processed, "
                    f"{len(candidates)} candidate(s) so far, {elapsed_text}"
                )

        qualifies = camera_result.success and len(camera_result.detected_ids) >= 3
        if not qualifies:
            _finalize_open_segment()
            boundary.reset()
            return

        starts_new = boundary.observe(camera_result.detected_ids, _stability_pose(camera_result))
        if starts_new:
            _finalize_open_segment()

        if segment_start_s is None:
            segment_start_s = timestamp_s
        segment_end_s = timestamp_s

        score = _frame_quality_score(camera_result)
        if best_in_segment is None or score > best_in_segment[0]:
            best_in_segment = (score, timestamp_s, camera_result, image.copy())

    def _cancel_progress() -> None:
        _finalize_open_segment()
        if progress_callback is not None:
            progress_callback(f"Marker extraction cancelled after {processed} frame(s) scanned.")

    detector_workers = max(1, min(int(detector_workers or 1), os.cpu_count() or 1))
    if detector_workers > 1:
        pending = deque()
        max_pending = detector_workers * 2
        cancelled = False

        def _detect(image: np.ndarray) -> CameraDetectionResult:
            return detect_camera_target(image, intrinsics, target)

        with ThreadPoolExecutor(max_workers=detector_workers) as executor:
            for image, timestamp_s, _frame_id in frames_factory():
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    break
                pending.append((image, timestamp_s, executor.submit(_detect, image)))
                while len(pending) >= max_pending:
                    pending_image, pending_timestamp_s, future = pending.popleft()
                    _process_detection(pending_image, pending_timestamp_s, future.result())
            while pending:
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    break
                pending_image, pending_timestamp_s, future = pending.popleft()
                _process_detection(pending_image, pending_timestamp_s, future.result())

        if cancelled:
            _cancel_progress()
        else:
            _finalize_open_segment()
    else:
        for image, timestamp_s, _frame_id in frames_factory():
            if cancel_check is not None and cancel_check():
                _cancel_progress()
                break

            camera_result = detect_camera_target(image, intrinsics, target)
            _process_detection(image, timestamp_s, camera_result)
        else:
            # Loop completed without a cancel-triggered break -- flush whatever
            # segment was still open at end of stream.
            _finalize_open_segment()

    summary = tracker.finalize(segment_count, candidates)
    summary.lidar_pairing_deferred = not pair_lidar

    if progress_callback is not None:
        elapsed_s = time.monotonic() - start_monotonic
        progress_callback(
            f"Marker extraction complete: {len(candidates)} candidate scene(s) found "
            f"in {elapsed_s:.1f}s."
        )

    return candidates, summary
