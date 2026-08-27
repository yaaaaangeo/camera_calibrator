"""
tests/test_camera_lidar_extraction_diagnostics.py
======================================================

camera_lidar.extraction_diagnostics: ExtractionDiagnosticTracker's funnel
counts, and diagnose()'s Case A-D auto-diagnosis text -- exercised with
synthetic CameraDetectionResult/SceneCandidate objects, mirroring
tests/test_camera_lidar_scene_extraction.py's fixture style.
"""

from __future__ import annotations

import numpy as np

from camera_lidar.camera_detector import CameraDetectionResult
from camera_lidar.extraction_diagnostics import ExtractionDiagnosticTracker, diagnose
from camera_lidar.types import SceneCandidate, SceneType


def _result(detected_ids=(), raw_ids=None, dictionary="DICT_4X4_50", success=True) -> CameraDetectionResult:
    raw_ids = list(raw_ids) if raw_ids is not None else list(range(len(detected_ids)))
    return CameraDetectionResult(
        success=success,
        detected_ids=frozenset(detected_ids),
        detected_marker_ids=raw_ids,
        markers_detected=len(detected_ids),
        markers_expected=4,
        dictionary_name=dictionary,
    )


def _candidate(scene_type: SceneType, has_cloud: bool = True) -> SceneCandidate:
    return SceneCandidate(
        candidate_id="c", segment_start_s=0.0, segment_end_s=1.0, representative_timestamp_s=0.5,
        camera_topic="/cam", lidar_topic="/lidar", image=np.zeros((2, 2, 3), dtype=np.uint8),
        camera_detection=None, scene_type=scene_type,
        cloud_points=np.zeros((5, 3), dtype=np.float32) if has_cloud else None,
    )


# ----------------------------------------------------------------------
# ExtractionDiagnosticTracker.observe
# ----------------------------------------------------------------------

def test_tracker_observe_counts_full_partial_invalid_and_raw_aruco():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result(("top_left", "top_right", "bottom_right", "bottom_left")))  # FULL
    tracker.observe(_result(("top_left", "top_right", "bottom_right")))                 # PARTIAL
    tracker.observe(_result(("top_left", "top_right")))                                 # INVALID (>=1, <3)
    tracker.observe(_result(raw_ids=[99]))  # raw ArUco seen but none match expected ids
    tracker.observe(_result())              # no raw ArUco at all

    s = tracker.summary
    assert s.total_frames == 5
    assert s.decoded_frames == 5
    assert s.frames_with_raw_aruco == 4  # all but the last
    assert s.frames_with_expected_ids == 3  # full + partial + invalid
    assert s.full_frames == 1
    assert s.partial_frames == 1
    assert s.invalid_frames == 1
    assert s.dictionary == "DICT_4X4_50"


def test_tracker_finalize_counts_segments_scene_types_and_lidar_pairing():
    tracker = ExtractionDiagnosticTracker()
    candidates = [
        _candidate(SceneType.VALID_FULL, has_cloud=True),
        _candidate(SceneType.VALID_FULL, has_cloud=False),
        _candidate(SceneType.VALID_PARTIAL, has_cloud=True),
    ]
    summary = tracker.finalize(segment_count=4, candidates=candidates)
    assert summary.stable_segments == 4
    assert summary.final_scene_candidates == 3
    assert summary.full_scenes == 2
    assert summary.partial_scenes == 1
    assert summary.candidates_missing_lidar_pairing == 1


# ----------------------------------------------------------------------
# diagnose() -- Cases A-D + healthy
# ----------------------------------------------------------------------

def test_diagnose_case_a_no_raw_aruco_at_all():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result())
    tracker.observe(_result())
    text = diagnose(tracker.summary)
    assert "Detector / Dictionary / Image Quality" in text


def test_diagnose_case_b_raw_aruco_but_no_expected_match():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result(raw_ids=[10, 11, 12, 13]))
    text = diagnose(tracker.summary)
    assert "never matched" in text
    assert "Expected Marker IDs" in text


def test_diagnose_case_c_expected_ids_seen_but_no_stable_segments():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result(("top_left", "top_right", "bottom_right")))
    summary = tracker.finalize(segment_count=0, candidates=[])
    text = diagnose(summary)
    assert "Scene Segmentation" in text or "Stability" in text


def test_diagnose_case_d_segments_found_but_no_final_candidates():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result(("top_left", "top_right", "bottom_right")))
    summary = tracker.finalize(segment_count=1, candidates=[])
    text = diagnose(summary)
    assert "Marker Count Classification" in text


def test_diagnose_all_candidates_missing_lidar_pairing():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result(("top_left", "top_right", "bottom_right", "bottom_left")))
    summary = tracker.finalize(segment_count=1, candidates=[_candidate(SceneType.VALID_FULL, has_cloud=False)])
    text = diagnose(summary)
    assert "LiDAR" in text


def test_diagnose_healthy_case():
    tracker = ExtractionDiagnosticTracker()
    tracker.observe(_result(("top_left", "top_right", "bottom_right", "bottom_left")))
    summary = tracker.finalize(segment_count=1, candidates=[_candidate(SceneType.VALID_FULL, has_cloud=True)])
    text = diagnose(summary)
    assert "1 candidate" in text
    assert "1 FULL" in text
