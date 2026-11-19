from __future__ import annotations

import json

import numpy as np

from calibration.recommender import compute_final_result
from calibration.types import (
    CalibrationResult,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameQuality,
    ImageInfo,
    PatternConfig,
    PatternType,
    QualityGrade,
    ValidationResult,
)
from export.json_export import build_export_dict
from export.report import generate_html_report


def _dataset() -> Dataset:
    return Dataset(
        frames=[
            Frame(
                image_info=ImageInfo("ok", "", 640, 480),
                detection=DetectionResult("ok", True, num_corners=24, board_area_ratio=0.25, board_tilt_deg=18.0),
                quality=FrameQuality(overall_score=91.0, grade=QualityGrade.EXCELLENT, blur_score=92.0),
            ),
            Frame(
                image_info=ImageInfo("fail", "", 640, 480),
                detection=DetectionResult("fail", False, failure_reason="no_corners"),
            ),
        ]
    )


def test_enhanced_html_and_json_report_sections(camera_config):
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)
    cal = CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.eye(3),
        distortion=np.zeros((5, 1)),
        rms_error=0.2,
        success=True,
    )
    val = ValidationResult(train_rms=0.2, test_rms=0.3, edge_rms=0.4, straightness_residual=0.2, success=True)
    calibration_results = {CameraModelType.PINHOLE: cal}
    validation_results = {CameraModelType.PINHOLE: val}
    final = compute_final_result(CameraModelType.PINHOLE, calibration_results, validation_results)

    html = generate_html_report("enhanced", camera_config, pattern, _dataset(), calibration_results, validation_results, final)
    payload = build_export_dict(
        camera_config, pattern, _dataset(), calibration_results, validation_results,
        CameraModelType.PINHOLE, final_result=final,
    )

    for expected in [
        "Dataset Quality &amp; Detection Statistics",
        "Cross Validation",
        "Bootstrap Stability",
        "Final Calibration Summary",
    ]:
        assert expected in html
    assert payload["dataset"]["failure_reasons"] == {"no_corners": 1}
    assert "cross_validation" in payload and "holdout" in payload["cross_validation"]
    assert "bootstrap_stability" in payload
    assert payload["final_calibration_summary"]["chosen_model"] == CameraModelType.PINHOLE.value
    json.dumps(payload, default=str)
