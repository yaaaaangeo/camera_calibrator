"""
tests/test_information_criteria.py
==================================

설계 문서 24/25번 - AIC/BIC 계산과 노출.
"""

from __future__ import annotations

import math

from calibration.recommender import (
    build_recommendation_message,
    compute_information_criteria,
    compute_model_scores,
    format_score_table,
    parameter_count_for_model,
)
from calibration.types import (
    CalibrationResult,
    CalibrationMethod,
    CameraConfig,
    CameraModelType,
    Dataset,
    FinalResult,
    PatternConfig,
    PatternType,
    QualityGrade,
    ResidualStats,
    ValidationResult,
)
from export.json_export import build_export_dict


def _cal(model: CameraModelType, rmse: float, n: int = 100) -> CalibrationResult:
    return CalibrationResult(
        model_name=model,
        rms_error=rmse,
        residual_stats=ResidualStats(n=n, rmse=rmse),
        success=True,
    )


def _val(rms: float) -> ValidationResult:
    return ValidationResult(test_rms=rms, edge_rms=rms, straightness_residual=rms, success=True)


def _val_with_p95(rms: float, p95: float) -> ValidationResult:
    return ValidationResult(
        test_rms=rms,
        edge_rms=rms,
        straightness_residual=rms,
        test_residual_stats=ResidualStats(n=100, rmse=rms, p95=p95),
        success=True,
    )


def test_information_criteria_formula_matches_design_doc():
    rss, n, k = 25.0, 100, 9

    aic, bic = compute_information_criteria(
        residual_sum_squares=rss,
        num_observations=n,
        parameter_count=k,
    )

    assert aic == 2 * k + n * math.log(rss / n)
    assert bic == k * math.log(n) + n * math.log(rss / n)


def test_parameter_counts_include_intrinsics_plus_model_distortion_params():
    assert parameter_count_for_model(CameraModelType.PINHOLE) == 4
    assert parameter_count_for_model(CameraModelType.BROWN_CONRADY) == 9
    assert parameter_count_for_model(CameraModelType.EXTENDED_PINHOLE) == 12
    assert parameter_count_for_model(CameraModelType.FISHEYE) == 8
    assert parameter_count_for_model(CameraModelType.EXTENDED_PINHOLE, use_rational_model=True) == 12


def test_compute_model_scores_populates_aic_bic_and_raw_inputs():
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, rmse=0.5, n=100),
        CameraModelType.BROWN_CONRADY: _cal(CameraModelType.BROWN_CONRADY, rmse=0.42, n=100),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, rmse=0.4, n=100),
    }
    validation_results = {
        CameraModelType.PINHOLE: _val(0.5),
        CameraModelType.BROWN_CONRADY: _val(0.42),
        CameraModelType.EXTENDED_PINHOLE: _val(0.4),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    by_model = {s.model_name: s for s in scores}

    pinhole = by_model[CameraModelType.PINHOLE]
    brown = by_model[CameraModelType.BROWN_CONRADY]
    extended = by_model[CameraModelType.EXTENDED_PINHOLE]
    assert pinhole.parameter_count == 4
    assert pinhole.num_observations == 100
    assert pinhole.residual_sum_squares == 25.0
    assert pinhole.aic is not None and pinhole.bic is not None
    assert brown.parameter_count == 9
    assert extended.parameter_count == 12
    assert extended.aic < pinhole.aic
    assert extended.bic < pinhole.bic


def test_object_releasing_result_is_excluded_from_model_scores():
    ro = _cal(CameraModelType.BROWN_CONRADY, rmse=0.1)
    ro.calibration_method = CalibrationMethod.OBJECT_RELEASING
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, rmse=0.5),
        CameraModelType.BROWN_CONRADY: ro,
    }
    validation_results = {
        CameraModelType.PINHOLE: _val(0.5),
        CameraModelType.BROWN_CONRADY: _val(0.1),
    }

    scores = compute_model_scores(calibration_results, validation_results)

    assert [s.model_name for s in scores] == [CameraModelType.PINHOLE]


def test_format_score_table_includes_aic_and_bic_rows():
    calibration_results = {CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, rmse=0.5)}
    validation_results = {CameraModelType.PINHOLE: _val(0.5)}
    scores = compute_model_scores(calibration_results, validation_results)

    text = format_score_table(scores, calibration_results, validation_results)

    assert "AIC" in text
    assert "BIC" in text


def test_json_export_includes_information_criteria_in_model_scores():
    model = CameraModelType.PINHOLE
    calibration_results = {model: _cal(model, rmse=0.5)}
    validation_results = {model: _val(0.5)}
    scores = compute_model_scores(calibration_results, validation_results)
    camera = CameraConfig(width=640, height=480)
    pattern = PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=7,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_5X5_100",
    )
    final = FinalResult(
        chosen_model=model,
        calibration=calibration_results[model],
        validation=validation_results[model],
        overall_grade=QualityGrade.GOOD,
        model_scores=scores,
    )

    payload = build_export_dict(
        camera, pattern, Dataset(), calibration_results, validation_results,
        model, final_result=final, model_scores=scores,
    )

    exported_score = payload["model_scores"][0]
    assert exported_score["parameter_count"] == 4
    assert exported_score["residual_sum_squares"] == 25.0
    assert exported_score["num_observations"] == 100
    assert exported_score["aic"] is not None
    assert exported_score["bic"] is not None
    assert exported_score["selection_confidence"] == 100.0
    assert exported_score["selection_confidence_level"] == "HIGH"
    assert exported_score["selection_confidence_reason"] == "Only one model calibrated successfully."


def test_close_models_get_low_selection_confidence_warning():
    calibration_results = {
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, rmse=0.40),
        CameraModelType.FISHEYE: _cal(CameraModelType.FISHEYE, rmse=0.401),
    }
    validation_results = {
        CameraModelType.EXTENDED_PINHOLE: _val_with_p95(0.40, 0.72),
        CameraModelType.FISHEYE: _val_with_p95(0.401, 0.721),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    recommended = next(s for s in scores if s.is_recommended)
    message = build_recommendation_message(scores, calibration_results, validation_results)

    assert recommended.selection_confidence_level == "LOW"
    assert recommended.selection_confidence is not None
    assert "Model selection confidence: LOW" in message
    assert "perform similarly" in recommended.selection_confidence_reason


def test_clear_score_gap_gets_high_selection_confidence():
    calibration_results = {
        CameraModelType.PINHOLE: _cal(CameraModelType.PINHOLE, rmse=0.80),
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, rmse=0.20),
    }
    validation_results = {
        CameraModelType.PINHOLE: _val_with_p95(0.80, 1.20),
        CameraModelType.EXTENDED_PINHOLE: _val_with_p95(0.20, 0.30),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    recommended = next(s for s in scores if s.is_recommended)

    assert recommended.model_name == CameraModelType.EXTENDED_PINHOLE
    assert recommended.selection_confidence_level == "HIGH"
    assert recommended.selection_confidence >= 80.0


def test_score_table_includes_selection_confidence_row():
    calibration_results = {
        CameraModelType.EXTENDED_PINHOLE: _cal(CameraModelType.EXTENDED_PINHOLE, rmse=0.40),
        CameraModelType.FISHEYE: _cal(CameraModelType.FISHEYE, rmse=0.401),
    }
    validation_results = {
        CameraModelType.EXTENDED_PINHOLE: _val_with_p95(0.40, 0.72),
        CameraModelType.FISHEYE: _val_with_p95(0.401, 0.721),
    }

    scores = compute_model_scores(calibration_results, validation_results)
    text = format_score_table(scores, calibration_results, validation_results)

    assert "Confidence" in text
    assert "LOW" in text
