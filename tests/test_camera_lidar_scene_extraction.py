"""
tests/test_camera_lidar_scene_extraction.py
================================================

camera_lidar.scene_extraction: Stable Scene Segment grouping, representative-
frame selection, camera-only FULL/PARTIAL/INVALID candidate classification,
and the build_scene_candidates() end-to-end orchestration -- exercised with
synthetic CameraDetectionResult objects / a fake frames_factory rather than
real ArUco images, mirroring how tests/test_camera_lidar_core.py exercises
correspondence/pipeline logic directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from camera_lidar.camera_detector import CameraDetectionResult
from camera_lidar.gates import StabilityThresholds
from camera_lidar.scene_extraction import (
    build_scene_candidates,
    classify_candidate,
    detect_stable_segments,
    select_representative,
)
from camera_lidar.target_config import CORNER_ORDER, TargetConfig
from camera_lidar.types import SceneType

_BOARD = TargetConfig().circle_centers_board_frame()  # (4,3), CORNER_ORDER


def _make_camera_result(
    detected_ids,
    offset=(0.0, 0.0, 0.0),
    reprojection_error=0.3,
    success=True,
) -> CameraDetectionResult:
    detected_ids = frozenset(detected_ids)
    centers = (_BOARD + np.array(offset)) if success else None
    return CameraDetectionResult(
        success=success,
        circle_centers=centers,
        detected_ids=detected_ids,
        markers_detected=len(detected_ids),
        markers_expected=4,
        reprojection_error_px=reprojection_error if success else None,
    )


# ----------------------------------------------------------------------
# classify_candidate
# ----------------------------------------------------------------------

def test_classify_candidate_full_partial_invalid_boundary():
    target = TargetConfig()
    assert classify_candidate(_make_camera_result(CORNER_ORDER), target) == SceneType.VALID_FULL
    assert classify_candidate(_make_camera_result(CORNER_ORDER[:3]), target) == SceneType.VALID_PARTIAL
    assert classify_candidate(_make_camera_result(CORNER_ORDER[:2]), target) == SceneType.INVALID
    assert classify_candidate(_make_camera_result([]), target) == SceneType.INVALID


# ----------------------------------------------------------------------
# detect_stable_segments
# ----------------------------------------------------------------------

def test_detect_stable_segments_groups_consecutive_stable_frames():
    observations = [
        (0.0, _make_camera_result(CORNER_ORDER, offset=(0.0, 0.0, 0.0))),
        (0.1, _make_camera_result(CORNER_ORDER, offset=(0.001, 0.0, 0.0))),   # tiny move -> same segment
        (0.2, _make_camera_result(CORNER_ORDER, offset=(0.05, 0.0, 0.0))),    # big jump -> new segment
        (0.3, _make_camera_result(CORNER_ORDER, offset=(0.05, 0.001, 0.0))),  # tiny move -> same segment as prior
        (0.4, _make_camera_result(CORNER_ORDER[:3], offset=(0.05, 0.001, 0.0))),  # one marker flickers out, pose unchanged -> stays in the same segment
        (0.5, _make_camera_result([], success=False)),  # failed detection -> breaks the segment
        (0.6, _make_camera_result(CORNER_ORDER, offset=(0.2, 0.0, 0.0))),     # resumes -> new segment
    ]
    segments = detect_stable_segments(observations)
    assert segments == [[0, 1], [2, 3, 4], [6]]


def test_detect_stable_segments_empty_input():
    assert detect_stable_segments([]) == []


def test_detect_stable_segments_respects_custom_thresholds():
    # A 3cm move is inside a loosened 5cm threshold -> stays one segment.
    observations = [
        (0.0, _make_camera_result(CORNER_ORDER, offset=(0.0, 0.0, 0.0))),
        (0.1, _make_camera_result(CORNER_ORDER, offset=(0.03, 0.0, 0.0))),
    ]
    loose = StabilityThresholds(max_position_change_m=0.05, max_normal_change_deg=3.0)
    assert detect_stable_segments(observations, loose) == [[0, 1]]
    assert detect_stable_segments(observations) == [[0], [1]]  # default 2cm threshold splits it


def test_detect_stable_segments_tolerates_single_frame_marker_flicker_and_recovery():
    """A marker can drop out for exactly one frame (motion blur/lighting)
    and come back while the physical target never moved -- must stay ONE
    segment, not three. Regression for a real user-reported bug: real bag
    footage produced 1582 segments from what should have been a handful of
    held poses, because the old strict single-previous-frame/exact-ids
    comparison treated every such blip as "the target moved"."""
    observations = [
        (0.0, _make_camera_result(CORNER_ORDER)),
        (0.1, _make_camera_result(CORNER_ORDER[:3])),  # one marker flickers out
        (0.2, _make_camera_result(CORNER_ORDER)),       # flickers back in
        (0.3, _make_camera_result(CORNER_ORDER)),
    ]
    assert detect_stable_segments(observations) == [[0, 1, 2, 3]]


def test_detect_stable_segments_still_breaks_on_a_genuinely_different_partial_id_set():
    """Two DIFFERENT 3-marker patterns (neither a subset nor a superset of
    each other) are not treated as the same flicker -- the tolerance above
    only covers a subset/superset relationship to the segment's reference
    ids, not any arbitrary partial-id change."""
    observations = [
        (0.0, _make_camera_result(("top_left", "top_right", "bottom_right"))),
        (0.1, _make_camera_result(("top_right", "bottom_right", "bottom_left"))),
    ]
    assert len(detect_stable_segments(observations)) == 2


# ----------------------------------------------------------------------
# select_representative
# ----------------------------------------------------------------------

def test_select_representative_picks_highest_quality_not_first():
    segment_observations = [
        (0.0, _make_camera_result(CORNER_ORDER, reprojection_error=1.0)),   # score 399.0
        (0.1, _make_camera_result(CORNER_ORDER, reprojection_error=0.2)),   # score 399.8 <- best
        (0.2, _make_camera_result(CORNER_ORDER[:3], reprojection_error=0.05)),  # score 299.95
    ]
    assert select_representative(segment_observations) == 1


# ----------------------------------------------------------------------
# build_scene_candidates -- end to end over a synthetic multi-segment stream
# ----------------------------------------------------------------------

def _fake_frame(index: int) -> np.ndarray:
    return np.full((2, 2, 3), index, dtype=np.uint8)


def _frames_factory_for(specs: list[tuple[float, str]]):
    def factory():
        for i, (t_sec, _label) in enumerate(specs):
            yield _fake_frame(i), t_sec, f"frame_{i}"
    return factory


def test_build_scene_candidates_end_to_end(monkeypatch):
    # index -> (timestamp, camera_result); mirrors a bag with:
    #   segment A (frames 0-1): two FULL frames, same pose -> merge, rep = frame 1 (lower error)
    #   segment B (frames 2-3): a FULL frame + a same-pose 3-marker flicker
    #     frame -> merge (the fix under test), rep = frame 2 (more markers wins)
    #   segment C (frame 4 alone): a genuinely new pose, PARTIAL (3 markers) the whole time
    #   frame 5: INVALID (2 markers) -- breaks the segment it interrupts, never its own candidate
    #   segment D (frame 6 alone): a new pose, FULL
    results_by_index = {
        0: _make_camera_result(CORNER_ORDER, offset=(0.0, 0.0, 0.0), reprojection_error=1.0),
        1: _make_camera_result(CORNER_ORDER, offset=(0.001, 0.0, 0.0), reprojection_error=0.1),
        2: _make_camera_result(CORNER_ORDER, offset=(0.10, 0.0, 0.0), reprojection_error=0.2),
        3: _make_camera_result(CORNER_ORDER[:3], offset=(0.10, 0.0, 0.0), reprojection_error=0.2),
        4: _make_camera_result(CORNER_ORDER[:3], offset=(0.20, 0.0, 0.0), reprojection_error=0.2),
        5: _make_camera_result(CORNER_ORDER[:2], offset=(0.20, 0.0, 0.0), reprojection_error=0.2),
        6: _make_camera_result(CORNER_ORDER, offset=(0.30, 0.0, 0.0), reprojection_error=0.2),
    }
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def fake_detect_camera_target(image, intrinsics, target):
        return results_by_index[int(image.flat[0])]

    monkeypatch.setattr("camera_lidar.scene_extraction.detect_camera_target", fake_detect_camera_target)

    def frames_factory():
        for i, t_sec in enumerate(timestamps):
            yield _fake_frame(i), t_sec, f"frame_{i}"

    def cloud_lookup(t_sec):
        return np.zeros((5, 3), dtype=np.float32), t_sec + 0.001

    candidates, summary = build_scene_candidates(
        frames_factory=frames_factory,
        camera_topic="/cam",
        lidar_topic="/lidar",
        intrinsics=object(),
        target=TargetConfig(),
        cloud_lookup=cloud_lookup,
    )

    assert summary.total_frames == 7
    assert summary.stable_segments == 4
    assert summary.final_scene_candidates == 4
    assert summary.full_scenes == 3
    assert summary.partial_scenes == 1
    assert summary.candidates_missing_lidar_pairing == 0
    assert summary.expected_marker_ids == sorted(TargetConfig().marker_ids.values())

    assert [c.scene_type for c in candidates] == [
        SceneType.VALID_FULL, SceneType.VALID_FULL, SceneType.VALID_PARTIAL, SceneType.VALID_FULL,
    ]
    # Segment A covers frames 0-1; its representative is frame 1 (lower reprojection error).
    assert candidates[0].representative_timestamp_s == pytest.approx(0.1)
    assert candidates[0].segment_start_s == pytest.approx(0.0)
    assert candidates[0].segment_end_s == pytest.approx(0.1)
    # Segment B covers frames 2-3 (flicker-merged); representative is frame 2 (more markers).
    assert candidates[1].representative_timestamp_s == pytest.approx(0.2)
    assert candidates[1].segment_start_s == pytest.approx(0.2)
    assert candidates[1].segment_end_s == pytest.approx(0.3)
    # Segment C (PARTIAL) is frame 4 alone.
    assert candidates[2].representative_timestamp_s == pytest.approx(0.4)
    assert candidates[2].missing_ids == frozenset(CORNER_ORDER) - frozenset(CORNER_ORDER[:3])
    # Segment D is frame 6 alone (frame 5 was INVALID and excluded entirely).
    assert candidates[3].representative_timestamp_s == pytest.approx(0.6)


def test_build_scene_candidates_cancel_during_pass_one_keeps_partial_results(monkeypatch):
    """cancel_check firing mid-scan must stop promptly and still build
    coherent (if partial) segments/candidates from whatever was scanned --
    nothing already found is thrown away, and no half-built candidate (e.g.
    missing its image) is ever produced."""
    results_by_index = {i: _make_camera_result(CORNER_ORDER, offset=(float(i), 0.0, 0.0)) for i in range(10)}
    timestamps = [float(i) for i in range(10)]

    def fake_detect_camera_target(image, intrinsics, target):
        return results_by_index[int(image.flat[0])]

    monkeypatch.setattr("camera_lidar.scene_extraction.detect_camera_target", fake_detect_camera_target)

    def frames_factory():
        for i, t_sec in enumerate(timestamps):
            yield _fake_frame(i), t_sec, f"frame_{i}"

    seen = {"n": 0}

    def cancel_after_3():
        seen["n"] += 1
        return seen["n"] > 3

    candidates, summary = build_scene_candidates(
        frames_factory=frames_factory,
        camera_topic="/cam",
        lidar_topic="/lidar",
        intrinsics=object(),
        target=TargetConfig(),
        cloud_lookup=lambda t_sec: (np.zeros((5, 3), dtype=np.float32), t_sec),
        cancel_check=cancel_after_3,
    )

    assert summary.total_frames <= 3  # stopped well before scanning all 10 frames
    for c in candidates:
        assert c.image is not None  # never a half-built candidate with a missing representative image

    for c in candidates:
        assert c.image is not None
        assert c.cloud_points is not None
        assert c.candidate_id


def test_build_scene_candidates_keeps_candidate_when_no_lidar_pairing_found(monkeypatch):
    result = _make_camera_result(CORNER_ORDER)
    monkeypatch.setattr("camera_lidar.scene_extraction.detect_camera_target", lambda image, intrinsics, target: result)

    def frames_factory():
        yield _fake_frame(0), 0.0, "frame_0"

    candidates, summary = build_scene_candidates(
        frames_factory=frames_factory,
        camera_topic="/cam",
        lidar_topic="/lidar",
        intrinsics=object(),
        target=TargetConfig(),
        cloud_lookup=lambda t_sec: None,
    )
    assert len(candidates) == 1
    assert candidates[0].cloud_points is None
    assert candidates[0].cloud_timestamp_s is None
    assert summary.candidates_missing_lidar_pairing == 1
