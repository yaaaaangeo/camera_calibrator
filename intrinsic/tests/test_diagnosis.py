from __future__ import annotations

import numpy as np

from calibration.diagnosis import diagnose_calibration, format_diagnosis_report
from calibration.project_io import project_from_dict, project_to_dict
from calibration.recommender import compute_final_result
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraModelType,
    CoverageCell,
    Dataset,
    DiagnosisSeverity,
    DiversityScores,
    ObservabilityReport,
    ParameterCorrelation,
    ParameterUncertainty,
    PatternConfig,
    PatternType,
    RadialBin,
    RadialErrorProfile,
    RegionalError,
    ValidationResult,
)
from export.json_export import build_export_dict
from export.report import generate_html_report


def _cal() -> CalibrationResult:
    return CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE,
        camera_matrix=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        distortion=np.zeros((5, 1)),
        rms_error=0.4,
        regional_error=RegionalError(center=0.3, left=1.2, right=1.1, top=0.9, bottom=1.3, corner=1.4),
        radial_bands=RadialErrorProfile(
            bins=[
                RadialBin(0, 100, p95_error=0.35, label="Center"),
                RadialBin(500, 600, p95_error=1.2, label="Corner"),
            ]
        ),
        param_uncertainty_bootstrap=ParameterUncertainty(
            fx_std=12.0,
            fy_std=4.0,
            overall_stability=72.0,
            method="bootstrap",
        ),
        observability=ObservabilityReport(
            parameter_labels=["fx", "fy", "cx", "cy", "k1"],
            jacobian_rows=200,
            jacobian_cols=5,
            num_points=100,
            singular_values=[100.0, 10.0, 1.0, 0.1, 1e-9],
            rank=4,
            condition_number=1e11,
            max_abs_correlation=0.991,
            top_correlations=[ParameterCorrelation("fx", "k1", 0.991)],
            warnings=["High condition number: 1e+11."],
        ),
        success=True,
    )


def test_diagnosis_maps_metrics_to_failure_patterns():
    cal = _cal()
    val = ValidationResult(test_rms=1.1, edge_rms=1.25, success=True)

    report = diagnose_calibration(cal, val, dataset_coverage_pct=42.0)
    codes = {p.code for p in report.patterns}

    assert "poor_coverage" in codes
    assert "edge_residual_high" in codes
    assert "radial_edge_pattern" in codes
    assert "train_test_gap" in codes
    assert "unstable_parameters" in codes
    assert "rank_deficient_observability" in codes
    assert "ill_conditioned_observability" in codes
    assert "high_parameter_correlation" in codes
    assert any(p.severity == DiagnosisSeverity.ERROR for p in report.patterns)


def _coverage_grid_with_missing_corners() -> list[CoverageCell]:
    cells = []
    for row in range(4):
        for col in range(4):
            missing = (row, col) in {(0, 0), (3, 0)}
            cells.append(
                CoverageCell(
                    row=row,
                    col=col,
                    corner_count=0 if missing else 20,
                    coverage_score=0.0 if missing else 1.0,
                )
            )
    return cells


def test_diagnosis_reports_specific_coverage_location_gaps():
    cal = CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.eye(3),
        distortion=np.zeros((5, 1)),
        rms_error=0.2,
        success=True,
    )

    report = diagnose_calibration(
        cal,
        ValidationResult(test_rms=0.22, edge_rms=0.25, success=True),
        dataset_coverage_pct=87.5,
        coverage_grid=_coverage_grid_with_missing_corners(),
    )
    pattern = next(p for p in report.patterns if p.code == "coverage_location_gaps")

    assert "Top-left LOW" in " ".join(pattern.evidence)
    assert "Bottom-left LOW" in " ".join(pattern.evidence)
    assert "Top-left" in pattern.recommendation


def test_diagnosis_recommends_next_capture_actions_from_coverage_and_diversity():
    cal = CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.eye(3),
        distortion=np.zeros((5, 1)),
        rms_error=0.2,
        success=True,
    )

    report = diagnose_calibration(
        cal,
        ValidationResult(test_rms=0.22, edge_rms=0.25, success=True),
        dataset_coverage_pct=87.5,
        coverage_grid=_coverage_grid_with_missing_corners(),
        diversity=DiversityScores(
            position_coverage=0.80,
            distance_diversity=0.20,
            rotation_diversity=0.25,
            edge_coverage=0.45,
        ),
    )
    text = format_diagnosis_report(report)
    actions = " ".join(r.action for r in report.capture_recommendations)
    titles = " ".join(r.title for r in report.capture_recommendations)

    assert "Add Upper-left board views" in titles
    assert "20-30 degrees" in actions
    assert "close-distance" in actions
    assert "Next capture recommendations" in text


def test_diagnosis_formats_no_major_failure_pattern():
    cal = CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.eye(3),
        distortion=np.zeros((5, 1)),
        rms_error=0.2,
        success=True,
    )

    report = diagnose_calibration(cal, ValidationResult(test_rms=0.22, edge_rms=0.25, success=True), 90.0)
    text = format_diagnosis_report(report)

    assert report.patterns[0].code == "no_major_failure_pattern"
    assert "[INFO]" in text
    assert "Proceed with export" in text


def test_final_result_export_report_and_project_include_diagnosis(camera_config):
    cal = _cal()
    val = ValidationResult(test_rms=1.1, edge_rms=1.25, success=True)
    calibration_results = {CameraModelType.EXTENDED_PINHOLE: cal}
    validation_results = {CameraModelType.EXTENDED_PINHOLE: val}
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)

    final = compute_final_result(
        CameraModelType.EXTENDED_PINHOLE,
        calibration_results,
        validation_results,
        dataset_coverage_pct=42.0,
        coverage_grid=_coverage_grid_with_missing_corners(),
        dataset_diversity=DiversityScores(
            position_coverage=0.80,
            distance_diversity=0.20,
            rotation_diversity=0.25,
            edge_coverage=0.45,
        ),
    )
    payload = build_export_dict(
        camera_config, pattern, Dataset(), calibration_results, validation_results,
        CameraModelType.EXTENDED_PINHOLE, final_result=final,
    )
    html = generate_html_report("diag", camera_config, pattern, Dataset(), calibration_results, validation_results, final)
    project = CalibrationProject(
        project_name="diag",
        camera_config=camera_config,
        pattern_config=pattern,
        calibration_results=calibration_results,
        validation_results=validation_results,
        final_result=final,
    )
    restored = project_from_dict(project_to_dict(project))

    assert final.diagnosis is not None
    assert payload["final_result"]["diagnosis"].patterns[0].code == "poor_coverage"
    assert "Diagnosis &amp; Recommendations" in html
    assert "Insufficient image coverage" in html
    assert "Top-left LOW" in html
    assert "Next Capture Recommendations" in html
    assert "Add Upper-left board views" in html
    assert "20-30 degrees" in html
    assert "close-distance" in html
    assert restored.final_result.diagnosis.patterns[0].code == "poor_coverage"
    assert any(
        p.code == "coverage_location_gaps"
        for p in restored.final_result.diagnosis.patterns
    )
    assert any(
        r.code == "capture_tilt_20_30"
        for r in restored.final_result.diagnosis.capture_recommendations
    )
