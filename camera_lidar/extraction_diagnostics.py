"""
camera_calibrator.camera_lidar.extraction_diagnostics
==========================================================

Marker Extraction Diagnostic Mode: turns "MARKER EXTRACTION found 0 Scenes"
into a stage-by-stage funnel (how many frames made it past each stage) plus
an auto-generated diagnosis of WHERE the funnel emptied out and WHY -- so the
user never has to guess whether the ArUco detector itself is broken, the
target/dictionary configuration is wrong, or Scene Extraction's own
segmentation/pairing logic is discarding otherwise-good detections.

Deliberately its own module, not folded into camera_detector.py or
scene_extraction.py: camera_detector.py stays ignorant of Scene Extraction
concepts (segments, candidates), and scene_extraction.py's own loop only
gets one added line (`tracker.observe(...)`) instead of inline counting
logic -- the same separation of concerns the feature spec asked for, without
restructuring the existing flat camera_lidar/ module layout into nested
packages.

This is the "aggregate diagnostics" scope: per-frame counts only, not a
retained per-frame log (that would need to hold a diagnostic record for
every frame of a potentially multi-thousand-frame bag scan -- deferred; see
the plan doc for the full Debug Scene Browser this stops short of).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from camera_lidar.camera_detector import CameraDetectionResult

_MIN_EXPECTED_FOR_PARTIAL = 3


@dataclass
class ExtractionDiagnosticSummary:
    total_frames: int = 0
    decoded_frames: int = 0
    frames_with_raw_aruco: int = 0        # >=1 raw ArUco marker of ANY id (before expected-id filtering)
    frames_with_expected_ids: int = 0     # >=1 marker matching an expected id
    full_frames: int = 0                  # all expected ids matched
    partial_frames: int = 0               # exactly one short of all expected ids
    invalid_frames: int = 0               # >=1 expected match but fewer than _MIN_EXPECTED_FOR_PARTIAL
    stable_segments: int = 0
    final_scene_candidates: int = 0
    full_scenes: int = 0
    partial_scenes: int = 0
    candidates_missing_lidar_pairing: int = 0
    lidar_pairing_deferred: bool = False
    dictionary: str = ""
    expected_marker_ids: list[int] = field(default_factory=list)


class ExtractionDiagnosticTracker:
    """Stateful per-scan accumulator. `observe()` is called once per frame
    during scene_extraction.build_scene_candidates' streaming scan;
    `finalize()` is called once at the end with the final segment count and
    candidates list (no re-detection, just counting)."""

    def __init__(self) -> None:
        self.summary = ExtractionDiagnosticSummary()

    def observe(self, camera_result: CameraDetectionResult) -> None:
        self.summary.total_frames += 1
        # image decode itself already happened before detect_camera_target
        # was called (the frame reached this tracker at all) -- see
        # scene_extraction.build_scene_candidates, which only calls
        # detect_camera_target on frames its frames_factory successfully
        # decoded. A frame that failed to decode never reaches observe().
        self.summary.decoded_frames += 1
        if not self.summary.dictionary and camera_result.dictionary_name:
            self.summary.dictionary = camera_result.dictionary_name

        if camera_result.detected_marker_ids:
            self.summary.frames_with_raw_aruco += 1

        matched_count = len(camera_result.detected_ids)
        expected_count = camera_result.markers_expected
        if matched_count > 0:
            self.summary.frames_with_expected_ids += 1
        if matched_count >= expected_count and expected_count > 0:
            self.summary.full_frames += 1
        elif matched_count == expected_count - 1:
            self.summary.partial_frames += 1
        elif matched_count > 0:
            self.summary.invalid_frames += 1

    def finalize(self, segment_count: int, candidates: list) -> ExtractionDiagnosticSummary:
        self.summary.stable_segments = segment_count
        self.summary.final_scene_candidates = len(candidates)
        for candidate in candidates:
            if candidate.scene_type.value == "valid_full":
                self.summary.full_scenes += 1
            elif candidate.scene_type.value == "valid_partial":
                self.summary.partial_scenes += 1
            if candidate.cloud_points is None:
                self.summary.candidates_missing_lidar_pairing += 1
        return self.summary


def format_extraction_diagnostics(summary: ExtractionDiagnosticSummary) -> str:
    lines = [
        "MARKER EXTRACTION DIAGNOSTICS",
        "-" * 40,
        "",
        "INPUT",
        f"  Total Frames          {summary.total_frames}",
        f"  Image Decode OK       {summary.decoded_frames}",
        "",
        "ARUCO CONFIGURATION",
        f"  Dictionary            {summary.dictionary or '(unknown)'}",
        f"  Expected IDs          {', '.join(str(i) for i in sorted(summary.expected_marker_ids)) or '(none)'}",
        "",
        "RAW ARUCO DETECTION",
        f"  Frames with Any ArUco {summary.frames_with_raw_aruco}",
        "",
        "TARGET ID MATCHING",
        f"  Frames w/ Expected ID {summary.frames_with_expected_ids}",
        "",
        "MARKER COUNT",
        f"  Full (4/4)             {summary.full_frames}",
        f"  Partial (3/4)          {summary.partial_frames}",
        f"  Invalid (<=2)          {summary.invalid_frames}",
        "",
        "SCENE SEGMENTATION",
        f"  Stable Segments        {summary.stable_segments}",
        "",
        "LIDAR PAIRING",
        (
            "  Status                Deferred until ADD SELECTED"
            if summary.lidar_pairing_deferred
            else f"  Candidates w/o Pairing {summary.candidates_missing_lidar_pairing}"
        ),
        "",
        "FINAL",
        f"  Scene Candidates       {summary.final_scene_candidates}",
        f"    FULL                 {summary.full_scenes}",
        f"    PARTIAL              {summary.partial_scenes}",
        "",
        diagnose(summary),
    ]
    return "\n".join(lines)


def diagnose(summary: ExtractionDiagnosticSummary) -> str:
    """Auto-diagnosis: WHERE the funnel emptied out and WHY, from the
    already-computed counts above -- never a bare 'Extraction Failed'."""
    if summary.total_frames == 0:
        return (
            "DIAGNOSIS\n\n"
            "No frames were scanned from the selected camera topic.\n\n"
            "Check the Bag/Topic selection in Input Source."
        )

    if summary.frames_with_raw_aruco == 0:
        return (
            "DIAGNOSIS\n\n"
            "ArUco markers were not detected in ANY frame.\n\n"
            "Likely area: Detector / Dictionary / Image Quality.\n"
            "Use TEST CURRENT FRAME to check raw detection and rejected\n"
            "candidate count on a single frame, and TEST DICTIONARIES to\n"
            "rule out a dictionary mismatch."
        )

    if summary.frames_with_expected_ids == 0:
        return (
            "DIAGNOSIS\n\n"
            "ArUco detection is working (markers were found), but detected\n"
            "marker IDs never matched the configured FAST-Calib target's\n"
            "expected IDs.\n\n"
            "Check: Expected Marker IDs (Target Geometry), ArUco Dictionary.\n"
            "Use TEST CURRENT FRAME to compare Raw IDs against Expected IDs."
        )

    if summary.stable_segments == 0:
        return (
            "DIAGNOSIS\n\n"
            f"ArUco detection is working. {summary.frames_with_expected_ids} frame(s)\n"
            "contained at least one expected marker, but no Stable Scene\n"
            "Segment was ever formed.\n\n"
            "Likely area: Scene Segmentation / Stability Threshold\n"
            "(the target may never have held still long enough, or\n"
            "consecutive frames kept losing/gaining a marker ID)."
        )

    if summary.final_scene_candidates == 0:
        return (
            "DIAGNOSIS\n\n"
            f"{summary.stable_segments} stable segment(s) were found, but none\n"
            "produced a FULL or PARTIAL scene candidate.\n\n"
            "Likely area: Marker Count Classification (all segments'\n"
            "representative frames matched too few expected markers)."
        )

    if (
        not summary.lidar_pairing_deferred
        and summary.candidates_missing_lidar_pairing == summary.final_scene_candidates
    ):
        return (
            "DIAGNOSIS\n\n"
            f"{summary.final_scene_candidates} camera scene candidate(s) were found,\n"
            "but NONE could be paired with a LiDAR frame.\n\n"
            "Likely area: LiDAR topic selection / bag time range / sync."
        )

    return (
        "DIAGNOSIS\n\n"
        "ArUco detection and Scene Extraction both completed normally --\n"
        f"{summary.final_scene_candidates} candidate(s) found "
        f"({summary.full_scenes} FULL, {summary.partial_scenes} PARTIAL)."
    )
