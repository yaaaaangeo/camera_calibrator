from __future__ import annotations

import numpy as np

from calibration.compare import format_comparison_table
from calibration.project_io import project_from_dict, project_to_dict
from calibration.recommender import compute_final_result
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraModelType,
    QualityGrade,
)
from calibration.undistortion_quality import evaluate_undistortion_quality
from export.report import generate_html_report


def _result(distortion: np.ndarray) -> CalibrationResult:
    return CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE,
        camera_matrix=np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]),
        distortion=distortion,
        rms_error=0.2,
        success=True,
    )


def test_zero_distortion_has_full_valid_undistortion_area():
    report = evaluate_undistortion_quality(_result(np.zeros((5, 1))), (320, 240))

    assert report is not None
    assert report.valid_pixel_ratio > 0.99
    assert report.black_border_ratio < 0.01
    assert report.roi_loss_ratio < 0.02
    assert report.quality_grade == QualityGrade.GOOD


def test_distorted_model_reports_border_roi_and_sample_black_pixels():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    report = evaluate_undistortion_quality(
        _result(np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])),
        (320, 240),
        sample_image=image,
        sample_frame_id="sample_01",
    )

    assert report is not None
    assert report.valid_pixel_ratio < 1.0
    assert report.black_border_ratio > 0.0
    assert report.roi_loss_ratio > 0.0
    assert report.undistorted_black_pixel_ratio is not None
    assert report.undistorted_black_pixel_ratio > 0.0
    assert report.sample_frame_id == "sample_01"
    assert report.warnings


def test_undistortion_quality_is_exported_in_report_project_and_comparison(camera_config, pattern_config):
    from calibration.types import Dataset

    cal = _result(np.zeros((5, 1)))
    cal.undistortion_quality = evaluate_undistortion_quality(cal, (camera_config.width, camera_config.height))
    calibration_results = {CameraModelType.EXTENDED_PINHOLE: cal}
    project = CalibrationProject(
        project_name="uq",
        camera_config=camera_config,
        pattern_config=pattern_config,
        calibration_results=calibration_results,
    )

    restored = project_from_dict(project_to_dict(project))
    final = compute_final_result(CameraModelType.EXTENDED_PINHOLE, calibration_results, {})
    html = generate_html_report(
        "uq", camera_config, pattern_config, Dataset(), calibration_results, {}, final
    )
    table = format_comparison_table([cal])

    restored_report = restored.calibration_results[CameraModelType.EXTENDED_PINHOLE].undistortion_quality
    assert restored_report is not None
    assert restored_report.valid_pixel_ratio > 0.99
    assert "Undistortion Quality" in html
    assert "Valid Pixel Ratio" in html
    assert "Valid Pixels" in table
    assert "ROI Loss" in table
