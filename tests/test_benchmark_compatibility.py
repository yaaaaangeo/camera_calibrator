from __future__ import annotations

import numpy as np

from calibration.benchmark_compatibility import (
    CompatibilitySeverity,
    validate_calibration_pair_compatibility,
    validate_single_calibration,
)
from calibration.calibration_io import StandardCalibration
from calibration.types import CameraModelType


K = np.array([[800.0, 0.0, 640.0], [0.0, 802.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.array([-0.2, 0.05, 0.001, -0.002, 0.01], dtype=np.float64).reshape(-1, 1)


def _cal(**overrides) -> StandardCalibration:
    values = {
        "label": "cal",
        "camera_matrix": K.copy(),
        "distortion": D.copy(),
        "model_name": CameraModelType.EXTENDED_PINHOLE,
        "distortion_model": "plumb_bob",
        "width": 1280,
        "height": 720,
    }
    values.update(overrides)
    return StandardCalibration(**values)


def _codes(report_or_issues):
    issues = report_or_issues.issues if hasattr(report_or_issues, "issues") else report_or_issues
    return {i.code for i in issues}


def test_compatible_pair_passes_without_issues():
    report = validate_calibration_pair_compatibility(_cal(label="reference"), _cal(label="candidate"))

    assert report.compatible is True
    assert report.status == "compatible"
    assert report.issues == []


def test_pair_reports_resolution_model_distortion_model_and_count_mismatch():
    reference = _cal(label="reference")
    candidate = _cal(
        label="candidate",
        width=640,
        height=480,
        model_name=CameraModelType.FISHEYE,
        distortion_model="equidistant",
        distortion=np.zeros((4, 1)),
    )

    report = validate_calibration_pair_compatibility(reference, candidate)

    assert report.compatible is False
    assert {
        "reference_candidate_resolution_mismatch",
        "different_camera_models",
        "different_distortion_models",
        "different_distortion_coefficient_count",
    }.issubset(_codes(report))


def test_validation_image_size_mismatch_is_error():
    report = validate_calibration_pair_compatibility(
        _cal(label="reference"),
        _cal(label="candidate"),
        validation_image_size=(640, 480),
    )

    assert report.compatible is False
    assert "validation_resolution_mismatch" in _codes(report)


def test_single_calibration_checks_shape_nan_and_bottom_row():
    bad_k = np.array([[800.0, 0.0, 640.0], [0.0, np.nan, 360.0], [0.0, 0.0, 2.0]])
    issues = validate_single_calibration(_cal(camera_matrix=bad_k), side="reference")

    assert "camera_matrix_non_finite" in _codes(issues)
    assert "camera_matrix_bottom_row" in _codes(issues)
    assert any(i.severity == CompatibilitySeverity.ERROR for i in issues)


def test_single_calibration_checks_model_specific_distortion_count():
    issues = validate_single_calibration(
        _cal(model_name=CameraModelType.FISHEYE, distortion_model="equidistant", distortion=np.zeros((5, 1))),
        side="reference",
    )

    assert "distortion_count_fisheye" in _codes(issues)


def test_single_calibration_checks_pinhole_nonzero_distortion():
    issues = validate_single_calibration(
        _cal(model_name=CameraModelType.PINHOLE, distortion_model="none", distortion=np.array([0.1]).reshape(-1, 1)),
        side="reference",
    )

    assert "pinhole_distortion_nonzero" in _codes(issues)


def test_single_calibration_warns_on_parameter_ranges():
    bad = _cal(
        camera_matrix=np.array([[10.0, 2.0, 4000.0], [0.0, 50000.0, -1000.0], [0.0, 0.0, 1.0]]),
        distortion=np.array([9.0, 0, 0, 0, 0]).reshape(-1, 1),
    )

    issues = validate_single_calibration(bad, side="candidate")
    codes = _codes(issues)

    assert "camera_matrix_skew_nonzero" in codes
    assert "fx_parameter_range" in codes
    assert "fy_parameter_range" in codes
    assert "cx_parameter_range" in codes
    assert "cy_parameter_range" in codes
    assert "distortion_parameter_range" in codes
    assert any(i.severity == CompatibilitySeverity.WARNING for i in issues)
