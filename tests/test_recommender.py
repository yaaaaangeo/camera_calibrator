"""
tests/test_recommender.py
==============================

설계 문서 12번 - 최종 등급(FinalResult.overall_grade) 산정 로직.
"여러 지표 중 가장 나쁜 걸 기준으로 종합 등급을 매긴다"는 원칙(낙관 편향
방지)이 실제로 지켜지는지 확인한다.
"""

from __future__ import annotations

from calibration.recommender import compute_final_result
from calibration.types import (
    CalibrationResult,
    CameraModelType,
    QualityGrade,
    ValidationResult,
)


def _cal(rms: float, success: bool = True) -> CalibrationResult:
    return CalibrationResult(model_name=CameraModelType.EXTENDED_PINHOLE, rms_error=rms, success=success)


def _val(test_rms=None, edge_rms=None, straightness=None) -> ValidationResult:
    return ValidationResult(
        train_frame_ids=[], test_frame_ids=[],
        test_rms=test_rms, edge_rms=edge_rms, straightness_residual=straightness, success=True,
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
