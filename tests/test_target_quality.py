"""
tests/test_target_quality.py
===================================

설계 문서 3-2번 - Calibration Target(보드) 품질 검사.
"""

from __future__ import annotations

from calibration.target_quality import (
    TargetQualitySeverity,
    evaluate_dataset_target_quality,
    evaluate_target_quality,
    format_target_quality_summary,
)
from calibration.types import DetectionResult, Frame, FrameStatus, ImageInfo, PatternConfig, PatternType


def _pattern() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO, squares_x=7, squares_y=5,
        square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
    )


def _good_detection() -> DetectionResult:
    return DetectionResult(
        image_id="good", success=True, num_corners=24,
        board_area_ratio=0.3, board_center_px=(960.0, 540.0), board_tilt_deg=5.0,
        corner_confidence=1.0, min_edge_margin_px=200.0,
    )


class TestEvaluateTargetQuality:
    def test_good_detection_has_no_issues(self):
        report = evaluate_target_quality(_good_detection(), _pattern())
        assert report.issues == []

    def test_failed_detection_is_error(self):
        det = DetectionResult(image_id="bad", success=False, failure_reason="검출 실패 테스트")
        report = evaluate_target_quality(det, _pattern())
        assert report.has_errors
        assert any(i.code == "detection_failed" for i in report.issues)

    def test_none_detection_is_error(self):
        report = evaluate_target_quality(None, _pattern())
        assert report.has_errors

    def test_too_few_corners_is_warning(self):
        det = _good_detection()
        det.num_corners = 3
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "too_few_corners" for i in report.issues)

    def test_low_corner_confidence_is_warning(self):
        det = _good_detection()
        det.corner_confidence = 0.3
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "low_corner_confidence" for i in report.issues)

    def test_corner_near_edge_is_warning(self):
        det = _good_detection()
        det.min_edge_margin_px = 8.0
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "corner_near_edge" for i in report.issues)

    def test_very_close_to_edge_is_cut_off_warning(self):
        det = _good_detection()
        det.min_edge_margin_px = 1.0
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "board_likely_cut_off" for i in report.issues)
        # cut-off와 near-edge 둘 다 뜨면 중복 경고이므로 하나만 떠야 함
        assert not any(i.code == "corner_near_edge" for i in report.issues)

    def test_board_too_small_is_warning(self):
        det = _good_detection()
        det.board_area_ratio = 0.01
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "board_too_small" for i in report.issues)

    def test_board_too_large_is_warning(self):
        det = _good_detection()
        det.board_area_ratio = 0.9
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "board_too_large" for i in report.issues)

    def test_board_area_suboptimal_but_not_extreme(self):
        det = _good_detection()
        det.board_area_ratio = 0.07  # sweet spot(0.10~0.55) 살짝 아래, 하지만 hard 하한(0.02)보다는 큼
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "board_area_suboptimal" for i in report.issues)
        assert not any(i.code == "board_too_small" for i in report.issues)

    def test_extreme_tilt_is_warning(self):
        det = _good_detection()
        det.board_tilt_deg = 80.0
        report = evaluate_target_quality(det, _pattern())
        assert any(i.code == "board_tilt_extreme" for i in report.issues)


class TestDatasetLevel:
    def test_evaluate_dataset_target_quality_flags_only_bad_frames(self):
        good_frame = Frame(
            image_info=ImageInfo(image_id="good", path="-", width=1920, height=1080),
            detection=_good_detection(), status=FrameStatus.DETECTED,
        )
        bad_det = _good_detection()
        bad_det.image_id = "bad"
        bad_det.board_area_ratio = 0.01
        bad_frame = Frame(
            image_info=ImageInfo(image_id="bad", path="-", width=1920, height=1080),
            detection=bad_det, status=FrameStatus.DETECTED,
        )
        reports = evaluate_dataset_target_quality([good_frame, bad_frame], _pattern())
        assert reports["good"].issues == []
        assert reports["bad"].issues != []

    def test_format_summary_mentions_flagged_images(self):
        bad_det = _good_detection()
        bad_det.board_area_ratio = 0.01
        reports = {"bad": evaluate_target_quality(bad_det, _pattern())}
        text = format_target_quality_summary(reports)
        assert "bad" in text

    def test_format_summary_handles_no_issues(self):
        reports = {"good": evaluate_target_quality(_good_detection(), _pattern())}
        text = format_target_quality_summary(reports)
        assert "경고 없음" in text
