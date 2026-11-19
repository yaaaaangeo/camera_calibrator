"""
tests/test_recommender.py
==============================

설계 문서 12번 - 최종 등급(FinalResult.overall_grade) 산정 로직.
"여러 지표 중 가장 나쁜 걸 기준으로 종합 등급을 매긴다"는 원칙(낙관 편향
방지)이 실제로 지켜지는지 확인한다.
"""

from __future__ import annotations

from calibration.recommender import (
    build_recommendation_message,
    compute_final_result,
    compute_model_scores,
    format_score_table,
)
from calibration.types import (
    CalibrationResult,
    CameraModelType,
    ModelScoreWeights,
    ObservabilityReport,
    QualityGrade,
    RegionalError,
    ResidualStats,
    ValidationResult,
)


def _cal(rms: float, success: bool = True, p95: float | None = None) -> CalibrationResult:
    stats = ResidualStats(n=100, rmse=rms, p95=p95) if p95 is not None else None
    return CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE,
        rms_error=rms,
        residual_stats=stats,
        success=success,
    )


def _val(test_rms=None, edge_rms=None, straightness=None) -> ValidationResult:
    return ValidationResult(
        train_frame_ids=["train-1"], test_frame_ids=["test-1"],
        test_rms=test_rms, edge_rms=edge_rms, straightness_residual=straightness,
        straightness_source="test" if straightness is not None else None,
        test_residual_stats=ResidualStats(n=10, rmse=test_rms, p95=(test_rms * 2.0 if test_rms is not None else None))
        if test_rms is not None else None,
        per_frame_error={"test-1": test_rms} if test_rms is not None else {},
        success=True,
    )


def _p95_only_weights() -> ModelScoreWeights:
    return ModelScoreWeights(
        w_train=0.0, w_test=0.0, w_edge=0.0, w_line=0.0, w_complexity=0.0,
        w_p95=1.0, w_radial=0.0, w_aic=0.0, w_bic=0.0, w_stability=0.0,
        w_observability=0.0,
    )


def test_all_excellent_metrics_give_excellent_grade():
    cal = {CameraModelType.EXTENDED_PINHOLE: _cal(0.2)}
    val = {CameraModelType.EXTENDED_PINHOLE: _val(test_rms=0.2, edge_rms=0.25, straightness=0.1)}

    final = compute_final_result(CameraModelType.EXTENDED_PINHOLE, cal, val)
    assert final.overall_grade == QualityGrade.EXCELLENT
    assert final.confidence is not None
    assert final.confidence.score >= 85.0
    assert final.confidence.level == "HIGH"


def test_one_bad_metric_drags_down_overall_grade():
    """Train RMS는 훌륭한데 Edge RMS가 나쁘면, 종합 등급은 Edge 기준을
    따라가야 한다 (설계 문서 3.1 - 가장 나쁜 지표가 종합 판단을 좌우해야
    낙관 편향이 없다).
    """
    cal = {CameraModelType.EXTENDED_PINHOLE: _cal(0.2)}  # Excellent 수준
    val = {CameraModelType.EXTENDED_PINHOLE: _val(test_rms=0.2, edge_rms=2.5, straightness=0.1)}  # Edge는 Poor

    final = compute_final_result(CameraModelType.EXTENDED_PINHOLE, cal, val)
    assert final.overall_grade == QualityGrade.POOR, (
        "Train RMS가 훌륭해도 Edge RMS가 나쁘면 종합 등급이 Poor여야 함"
    )
    assert final.confidence is not None
    assert final.confidence.score <= 49.0


def test_failed_calibration_gives_reject():
    cal = {CameraModelType.PINHOLE: _cal(None, success=False)}
    val = {}
    final = compute_final_result(CameraModelType.PINHOLE, cal, val)
    assert final.overall_grade == QualityGrade.REJECT
    assert final.confidence is not None
    assert final.confidence.score == 0.0
    assert final.confidence.level == "REJECT"


def test_chosen_model_not_recommended_model_is_respected():
    """설계 문서 8번 - '추천과 선택의 분리'. 사용자가 추천 모델이 아닌 다른
    모델을 선택했다면, FinalResult는 항상 '선택된' 모델 기준으로 나가야 한다.
    """
    cal = {
        CameraModelType.PINHOLE: _cal(0.9),
        CameraModelType.FISHEYE: _cal(0.2),
    }
    val = {
        CameraModelType.PINHOLE: _val(test_rms=0.9, edge_rms=0.9, straightness=0.4),
        CameraModelType.FISHEYE: _val(test_rms=0.2, edge_rms=0.2, straightness=0.1),
    }
    # Fisheye가 더 좋은 지표를 가졌지만, 사용자가 Pinhole을 선택했다고 가정
    final = compute_final_result(CameraModelType.PINHOLE, cal, val)
    assert final.chosen_model == CameraModelType.PINHOLE
    assert final.calibration is cal[CameraModelType.PINHOLE]


def test_missing_validation_falls_back_to_train_rms_only():
    """validation_results에 아직 값이 없어도(예: hold-out을 안 돌린 경우)
    최소한 train RMS 기준으로는 등급을 매겨야 한다 - 크래시하면 안 됨.
    """
    cal = {CameraModelType.PINHOLE: _cal(0.4)}
    final = compute_final_result(CameraModelType.PINHOLE, cal, {})
    assert final.overall_grade in (QualityGrade.VERY_GOOD, QualityGrade.GOOD)
    assert final.confidence is not None
    assert "Hold-out validation is missing or failed." in final.confidence.warnings


def test_model_with_missing_core_validation_is_not_selection_eligible():
    """A calibrated model without core hold-out residual stats is not a recommendation candidate."""
    cal = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE,
            rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4, p95=0.1),
            success=True,
        ),
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4, p95=10.0),
            success=True,
        ),
    }
    val = {
        CameraModelType.PINHOLE: ValidationResult(test_rms=0.5, success=True),
        CameraModelType.BROWN_CONRADY: ValidationResult(test_rms=0.5, success=True),
    }
    weights = ModelScoreWeights(
        w_train=0.0, w_test=0.0, w_edge=0.0, w_line=0.0, w_complexity=0.0,
        w_p95=1.0, w_radial=0.0, w_aic=0.0, w_bic=0.0, w_stability=0.0,
        w_observability=0.0,
    )

    scores = compute_model_scores(cal, val, weights)

    assert all(not s.is_selection_eligible for s in scores)
    assert all(s.selection_status == "VALIDATION INCOMPLETE" for s in scores)
    table = format_score_table(scores, cal, val)
    p95_line = next(line for line in table.splitlines() if line.startswith("Test P95"))
    score_line = next(line for line in table.splitlines() if line.startswith("Score"))
    assert p95_line.count("N/A") == 2
    assert score_line.count("Not eligible") == 2
    assert "0.100" not in p95_line
    assert "10.000" not in p95_line


def test_model_score_test_p95_does_not_fallback_to_train_residual_p95_when_stats_lack_p95():
    cal = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE,
            rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4, p95=0.1),
            success=True,
        ),
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4, p95=10.0),
            success=True,
        ),
    }
    val = {
        CameraModelType.PINHOLE: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"], test_rms=0.5,
            test_residual_stats=ResidualStats(n=100, rmse=0.5, p95=None),
            per_frame_error={"test-1": 0.5}, success=True,
        ),
        CameraModelType.BROWN_CONRADY: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"], test_rms=0.5,
            test_residual_stats=ResidualStats(n=100, rmse=0.5, p95=None),
            per_frame_error={"test-1": 0.5}, success=True,
        ),
    }

    scores = compute_model_scores(cal, val, weights=_p95_only_weights())
    table = format_score_table(scores, cal, val)
    p95_line = next(line for line in table.splitlines() if line.startswith("Test P95"))

    assert all(s.is_selection_eligible for s in scores)
    assert {s.components["p95"] for s in scores} == {0.0}
    assert p95_line.count("N/A") == 2
    assert "0.100" not in p95_line
    assert "10.000" not in p95_line


def test_validation_failed_model_is_excluded_from_recommendation_candidates():
    cal = {
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4), success=True,
        ),
        CameraModelType.EXTENDED_PINHOLE: CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE, rms_error=0.5,
            residual_stats=ResidualStats(n=100, rmse=0.5), success=True,
        ),
        CameraModelType.FISHEYE: CalibrationResult(
            model_name=CameraModelType.FISHEYE, rms_error=0.1,
            residual_stats=ResidualStats(n=100, rmse=0.1), success=True,
        ),
    }
    val = {
        CameraModelType.BROWN_CONRADY: _val(test_rms=0.4, edge_rms=0.4),
        CameraModelType.EXTENDED_PINHOLE: _val(test_rms=0.5, edge_rms=0.5),
        CameraModelType.FISHEYE: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"],
            success=False, error_message="solvePnP failed",
        ),
    }

    scores = compute_model_scores(cal, val)
    by_model = {s.model_name: s for s in scores}

    assert not by_model[CameraModelType.FISHEYE].is_selection_eligible
    assert by_model[CameraModelType.FISHEYE].selection_status == "VALIDATION INCOMPLETE"
    assert by_model[CameraModelType.FISHEYE].selection_confidence == 0.0
    assert not by_model[CameraModelType.FISHEYE].is_recommended
    assert sum(1 for s in scores if s.is_recommended) == 1
    assert next(s for s in scores if s.is_recommended).model_name != CameraModelType.FISHEYE


def test_test_edge_rms_does_not_fallback_to_train_regional_error():
    cal = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE,
            rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4),
            regional_error=RegionalError(corner=0.1),
            success=True,
        ),
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY,
            rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4),
            regional_error=RegionalError(corner=10.0),
            success=True,
        ),
    }
    val = {
        CameraModelType.PINHOLE: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"], test_rms=0.5,
            edge_rms=None, test_residual_stats=ResidualStats(n=100, rmse=0.5, p95=1.0),
            per_frame_error={"test-1": 0.5}, success=True,
        ),
        CameraModelType.BROWN_CONRADY: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"], test_rms=0.5,
            edge_rms=None, test_residual_stats=ResidualStats(n=100, rmse=0.5, p95=1.0),
            per_frame_error={"test-1": 0.5}, success=True,
        ),
    }
    weights = ModelScoreWeights(
        w_train=0.0, w_test=0.0, w_edge=1.0, w_line=0.0, w_complexity=0.0,
        w_p95=0.0, w_radial=0.0, w_aic=0.0, w_bic=0.0, w_stability=0.0,
        w_observability=0.0,
    )

    scores = compute_model_scores(cal, val, weights)
    table = format_score_table(scores, cal, val)
    edge_line = next(line for line in table.splitlines() if line.startswith("Edge RMS"))

    assert {s.components["edge"] for s in scores} == {0.0}
    assert edge_line.count("N/A") == 2
    assert "0.100" not in edge_line
    assert "10.000" not in edge_line


def test_train_fallback_straightness_is_not_positive_holdout_evidence():
    cal = {
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4), success=True,
        ),
        CameraModelType.EXTENDED_PINHOLE: CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4), success=True,
        ),
    }
    val = {
        CameraModelType.BROWN_CONRADY: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"], test_rms=0.4,
            straightness_residual=0.8, straightness_source="test",
            test_residual_stats=ResidualStats(n=100, rmse=0.4, p95=0.8),
            per_frame_error={"test-1": 0.4}, success=True,
        ),
        CameraModelType.EXTENDED_PINHOLE: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=["test-1"], test_rms=0.4,
            straightness_residual=0.5, straightness_source="train_fallback",
            test_residual_stats=ResidualStats(n=100, rmse=0.4, p95=0.8),
            per_frame_error={"test-1": 0.4}, success=True,
        ),
    }
    weights = ModelScoreWeights(
        w_train=0.0, w_test=0.0, w_edge=0.0, w_line=1.0, w_complexity=0.0,
        w_p95=0.0, w_radial=0.0, w_aic=0.0, w_bic=0.0, w_stability=0.0,
        w_observability=0.0,
    )

    scores = compute_model_scores(cal, val, weights)
    message = build_recommendation_message(scores, cal, val)

    assert {s.components["line"] for s in scores} == {0.0}
    assert "Best Test Straightness" not in message


def test_poor_observability_is_not_positive_selection_reason():
    cal = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4),
            observability=ObservabilityReport(
                jacobian_cols=4, rank=4, condition_number=1e12,
                max_abs_correlation=0.99, observability_score=0.0,
                observability_grade="POOR",
            ),
            success=True,
        ),
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY, rms_error=0.5,
            residual_stats=ResidualStats(n=100, rmse=0.5),
            observability=ObservabilityReport(
                jacobian_cols=9, rank=9, condition_number=1e12,
                max_abs_correlation=0.99, observability_score=0.0,
                observability_grade="POOR",
            ),
            success=True,
        ),
    }
    val = {
        CameraModelType.PINHOLE: _val(test_rms=0.4, edge_rms=0.4),
        CameraModelType.BROWN_CONRADY: _val(test_rms=0.5, edge_rms=0.5),
    }

    scores = compute_model_scores(cal, val)
    recommended = next(s for s in scores if s.is_recommended)

    assert all("Best Observability" not in reason for reason in recommended.selection_reasons)


def test_poor_observability_is_not_used_by_generic_fallback_reason():
    cal = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4),
            observability=ObservabilityReport(
                jacobian_cols=4, rank=4, condition_number=1e12,
                normalized_condition_number=1e12,
                max_abs_correlation=0.99, observability_score=0.0,
                observability_grade="POOR",
            ),
            success=True,
        ),
        CameraModelType.BROWN_CONRADY: CalibrationResult(
            model_name=CameraModelType.BROWN_CONRADY, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4),
            observability=ObservabilityReport(
                jacobian_cols=9, rank=9, condition_number=1e12,
                normalized_condition_number=1e12,
                max_abs_correlation=0.99, observability_score=0.0,
                observability_grade="POOR",
            ),
            success=True,
        ),
    }
    val = {
        CameraModelType.PINHOLE: _val(test_rms=0.4),
        CameraModelType.BROWN_CONRADY: _val(test_rms=0.4),
    }
    weights = ModelScoreWeights(
        w_train=0.0, w_test=0.0, w_edge=0.0, w_line=0.0, w_complexity=0.0,
        w_p95=0.0, w_radial=0.0, w_aic=0.0, w_bic=0.0, w_stability=0.0,
        w_observability=1.0,
    )

    scores = compute_model_scores(cal, val, weights)
    message = build_recommendation_message(scores, cal, val)

    assert "Best Observability" not in message
    assert "Low weighted Observability" not in message


def test_recommendation_message_distinguishes_validation_incomplete_from_calibration_failure():
    cal = {
        CameraModelType.PINHOLE: CalibrationResult(
            model_name=CameraModelType.PINHOLE, rms_error=0.4,
            residual_stats=ResidualStats(n=100, rmse=0.4), success=True,
        ),
    }
    val = {
        CameraModelType.PINHOLE: ValidationResult(
            train_frame_ids=["train-1"], test_frame_ids=[],
            success=True, error_message="No test split.",
        ),
    }

    scores = compute_model_scores(cal, val)
    message = build_recommendation_message(scores, cal, val)

    assert "캘리브레이션은 성공" in message
    assert "Hold-out validation evidence" in message
    assert "VALIDATION INCOMPLETE" in message
    assert "캘리브레이션이 실패" not in message
