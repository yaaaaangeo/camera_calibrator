from __future__ import annotations

import numpy as np

from calibration.recommender import build_recommendation_message, compute_model_scores, format_score_table
from calibration.types import (
    CalibrationResult,
    CameraModelType,
    ModelScoreWeights,
    ObservabilityReport,
    ParameterUncertainty,
    RadialBin,
    RadialErrorProfile,
    ResidualStats,
    ValidationResult,
)
from export.json_export import build_export_dict
from export.report import generate_html_report


def _cal(model: CameraModelType, rmse: float) -> CalibrationResult:
    return CalibrationResult(
        model_name=model,
        camera_matrix=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        distortion=np.zeros((5, 1)),
        rms_error=rmse,
        residual_stats=ResidualStats(n=100, rmse=rmse, p95=rmse * 2.0),
        radial_bands=RadialErrorProfile(
            bins=[
                RadialBin(0, 100, p95_error=rmse, label="Center"),
                RadialBin(400, 500, p95_error=rmse * 2.0, label="Edge"),
            ]
        ),
        param_uncertainty=ParameterUncertainty(overall_stability=95.0),
        observability=ObservabilityReport(
            jacobian_rows=200,
            jacobian_cols=4,
            rank=4,
            condition_number=1e5,
            max_abs_correlation=0.85,
        ),
        success=True,
    )


def _val(rms: float, p95: float) -> ValidationResult:
    return ValidationResult(
        test_rms=rms,
        edge_rms=rms,
        straightness_residual=rms,
        test_residual_stats=ResidualStats(n=50, rmse=rms, p95=p95),
        success=True,
    )


def _p95_only_weights() -> ModelScoreWeights:
    return ModelScoreWeights(
        w_train=0.0,
        w_test=0.0,
        w_edge=0.0,
        w_line=0.0,
        w_complexity=0.0,
        w_p95=1.0,
        w_radial=0.0,
        w_aic=0.0,
        w_bic=0.0,
        w_stability=0.0,
        w_observability=0.0,
    )


def test_p95_component_participates_in_model_ranking():
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, 0.30),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, 0.30),
    }
    validation_results = {
        CameraModelType.PINHOLE: _val(0.30, p95=1.20),
        CameraModelType.EXTENDED_PINHOLE: _val(0.30, p95=0.40),
    }

    scores = compute_model_scores(calibration_results, validation_results, weights=_p95_only_weights())
    recommended = next(s for s in scores if s.is_recommended)

    assert recommended.model_name == CameraModelType.EXTENDED_PINHOLE
    assert recommended.components["p95"] == 0.0
    assert next(s for s in scores if s.model_name == CameraModelType.PINHOLE).components["p95"] == 1.0


def test_default_score_exposes_new_metric_components_and_table_rows():
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, 0.50),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, 0.40),
    }
    validation_results = {
        CameraModelType.PINHOLE: _val(0.55, p95=1.10),
        CameraModelType.EXTENDED_PINHOLE: _val(0.45, p95=0.80),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    text = format_score_table(scores, calibration_results, validation_results)

    for score in scores:
        for key in ("p95", "radial", "aic", "bic", "stability", "observability"):
            assert key in score.components
    for label in ("Test P95", "Radial", "AIC", "BIC", "Stability", "Obs Penalty"):
        assert label in text


def test_confidence_is_populated_for_each_successful_model():
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, 0.60),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, 0.35),
        CameraModelType.FISHEYE: _cal(CameraModelType.FISHEYE, 0.45),
    }
    validation_results = {
        CameraModelType.PINHOLE: _val(0.60, p95=1.20),
        CameraModelType.EXTENDED_PINHOLE: _val(0.35, p95=0.55),
        CameraModelType.FISHEYE: _val(0.45, p95=0.80),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    confidences = [s.selection_confidence for s in scores]
    text = format_score_table(scores, calibration_results, validation_results)

    assert all(c is not None for c in confidences)
    assert round(sum(confidences), 1) == 100.0
    assert all(s.selection_confidence_level in ("LOW", "MEDIUM", "HIGH") for s in scores)
    assert text.count("%") == 3


def test_failed_model_gets_zero_confidence():
    calibration_results = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE,
            rms_error=None,
            success=False,
            error_message="boom",
        ),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, 0.35),
    }
    validation_results = {
        CameraModelType.EXTENDED_PINHOLE: _val(0.35, p95=0.55),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    failed = next(s for s in scores if s.model_name == CameraModelType.PINHOLE)

    assert failed.selection_confidence == 0.0
    assert failed.selection_confidence_level == "LOW"
    assert "Calibration failed" in failed.selection_confidence_reason


def test_recommendation_reasons_explain_best_extended_metrics(camera_config):
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, 0.55),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, 0.30),
        CameraModelType.FISHEYE: _cal(CameraModelType.FISHEYE, 0.45),
    }
    validation_results = {
        CameraModelType.PINHOLE: _val(0.55, p95=1.40),
        CameraModelType.EXTENDED_PINHOLE: _val(0.30, p95=0.45),
        CameraModelType.FISHEYE: _val(0.45, p95=0.90),
    }
    # Make the recommended model clearly best in radial/stability/observability too.
    calibration_results[CameraModelType.PINHOLE].radial_bands.bins[1].p95_error = 1.6
    calibration_results[CameraModelType.EXTENDED_PINHOLE].radial_bands.bins[1].p95_error = 0.35
    calibration_results[CameraModelType.FISHEYE].radial_bands.bins[1].p95_error = 1.0
    calibration_results[CameraModelType.EXTENDED_PINHOLE].param_uncertainty.overall_stability = 98.0
    calibration_results[CameraModelType.PINHOLE].param_uncertainty.overall_stability = 91.0
    calibration_results[CameraModelType.FISHEYE].param_uncertainty.overall_stability = 93.0
    calibration_results[CameraModelType.EXTENDED_PINHOLE].observability.condition_number = 1e4
    calibration_results[CameraModelType.PINHOLE].observability.condition_number = 1e8
    calibration_results[CameraModelType.FISHEYE].observability.condition_number = 1e7

    scores = compute_model_scores(calibration_results, validation_results)
    recommended = next(s for s in scores if s.is_recommended)
    message = build_recommendation_message(scores, calibration_results, validation_results)

    assert recommended.model_name == CameraModelType.EXTENDED_PINHOLE
    assert any("Lowest Test P95" in r for r in recommended.selection_reasons)
    assert any("Best Radial Error" in r for r in recommended.selection_reasons)
    assert any("Best BIC" in r for r in recommended.selection_reasons)
    assert any("Stable Parameters" in r or "Most Stable Parameters" in r for r in recommended.selection_reasons)
    assert "Reasons:" in message
    assert "✓ Lowest Test P95" in message

    from calibration.types import Dataset, FinalResult, PatternConfig, PatternType, QualityGrade

    final = FinalResult(
        chosen_model=CameraModelType.EXTENDED_PINHOLE,
        calibration=calibration_results[CameraModelType.EXTENDED_PINHOLE],
        validation=validation_results[CameraModelType.EXTENDED_PINHOLE],
        overall_grade=QualityGrade.GOOD,
        model_scores=scores,
    )
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)
    payload = build_export_dict(
        camera_config, pattern, Dataset(), calibration_results, validation_results,
        CameraModelType.EXTENDED_PINHOLE, final_result=final, model_scores=scores,
    )
    html = generate_html_report(
        "reasons", camera_config, pattern, Dataset(), calibration_results, validation_results, final
    )

    exported = next(s for s in payload["model_scores"] if s["is_recommended"])
    assert any("Lowest Test P95" in r for r in exported["selection_reasons"])
    assert "Recommendation Reasons" in html
    assert "Best Radial Error" in html
