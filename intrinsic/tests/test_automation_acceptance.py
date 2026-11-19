"""automation.acceptance: PASS / WARN / FAIL 판정 규칙.

임계값 숫자는 이 테스트 안에서만 쓰는 예시 값이다 (운영 기준 아님).
"""

from __future__ import annotations

import pytest

from automation.acceptance import (
    FAIL,
    METRICS,
    PASS,
    STATUS_NOT_AVAILABLE,
    STATUS_NOT_CONFIGURED,
    WARN,
    AcceptanceEvidence,
    AcceptancePolicyError,
    evaluate_acceptance,
    parse_policy,
)

GOOD_METRICS = {
    "train_rms_px": 0.25,
    "test_rms_px": 0.30,
    "test_p95_px": 0.55,
    "test_p99_px": 0.60,
    "edge_rms_px": 0.27,
    "train_test_gap_px": 0.03,
    "stability_pct": 99.0,
    "observability_score": 80.0,
    "coverage_pct": 100.0,
    "num_images_used": 24.0,
    "num_test_frames": 6.0,
}

EXAMPLE_LIMITS = {
    "train_rms_px": 0.5,
    "test_rms_px": 0.5,
    "test_p95_px": 1.0,
    "edge_rms_px": 0.6,
    "coverage_pct": 75.0,
    "num_images_used": 15,
}


def _evidence(**overrides) -> AcceptanceEvidence:
    data = dict(
        chosen_model="brown_conrady",
        recommended_model="brown_conrady",
        calibration_success=True,
        validation_success=True,
        train_frame_ids=["a", "b", "c"],
        test_frame_ids=["d", "e"],
        metrics=dict(GOOD_METRICS),
        overall_grade="excellent",
        confidence_level="HIGH",
        confidence_score=90.0,
        holdout_evidence_status="sufficient",
    )
    data.update(overrides)
    return AcceptanceEvidence(**data)


def _policy(limits: dict | None = None, required: set[str] | None = None, **extra):
    limits = EXAMPLE_LIMITS if limits is None else limits
    required = set(limits) if required is None else required
    thresholds = {
        name: {METRICS[name].direction: limits.get(name), "required": name in required}
        for name in set(limits) | required
    }
    return parse_policy({"thresholds": thresholds, **extra})


def _statuses(result) -> dict[str, str]:
    return {c.check_id: c.status for c in result.checks}


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_required_metric_is_missing_and_criteria_incomplete(value):
    evidence = _evidence()
    evidence.metrics["coverage_pct"] = value
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == FAIL
    assert result.criteria_complete is False
    check = next(c for c in result.checks if c.metric == "coverage_pct")
    assert check.kind == "missing_metric"
    assert check.value is None


def test_pass_when_all_required_thresholds_met_and_no_existing_warnings():
    result = evaluate_acceptance(_evidence(), _policy())
    assert result.verdict == PASS
    assert result.criteria_complete is True
    assert result.recommended_model == "brown_conrady"


def test_unconfigured_required_thresholds_cannot_pass():
    result = evaluate_acceptance(_evidence(), _policy(limits={k: None for k in EXAMPLE_LIMITS}))
    assert result.verdict == WARN
    assert result.criteria_complete is False
    assert any("운영 기준이 정해지지 않아" in r for r in result.reasons)
    assert _statuses(result)["threshold:test_rms_px"] == STATUS_NOT_CONFIGURED


def test_no_required_criteria_at_all_cannot_pass():
    result = evaluate_acceptance(_evidence(), parse_policy({}))
    assert result.verdict == WARN
    assert result.criteria_complete is False


def test_threshold_violation_fails_by_default():
    evidence = _evidence(metrics=GOOD_METRICS | {"test_rms_px": 0.9})
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == FAIL
    assert any("test_rms_px" in r and "기준 미달" in r for r in result.reasons)


def test_threshold_violation_can_be_configured_as_warning():
    policy = parse_policy({"thresholds": {
        name: {METRICS[name].direction: value, "required": True,
               **({"on_violation": "warn"} if name == "edge_rms_px" else {})}
        for name, value in EXAMPLE_LIMITS.items()
    }})
    result = evaluate_acceptance(_evidence(metrics=GOOD_METRICS | {"edge_rms_px": 0.8}), policy)
    assert result.verdict == WARN


def test_min_direction_metric_violation():
    result = evaluate_acceptance(_evidence(metrics=GOOD_METRICS | {"coverage_pct": 40.0}), _policy())
    assert result.verdict == FAIL
    assert _statuses(result)["threshold:coverage_pct"] == "fail"


def test_missing_required_metric_fails_instead_of_assuming_good_value():
    evidence = _evidence(metrics=GOOD_METRICS | {"test_p95_px": None})
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == FAIL
    assert any("test_p95_px" in r and "계산되지 않았습니다" in r for r in result.reasons)


def test_missing_optional_metric_is_recorded_but_does_not_block_pass():
    limits = EXAMPLE_LIMITS | {"stability_pct": 80.0}
    policy = _policy(limits=limits, required=set(EXAMPLE_LIMITS))
    result = evaluate_acceptance(_evidence(metrics=GOOD_METRICS | {"stability_pct": None}), policy)
    assert result.verdict == PASS
    assert _statuses(result)["threshold:stability_pct"] == STATUS_NOT_AVAILABLE


def test_calculation_failure_is_fail_and_skips_other_checks():
    evidence = _evidence(calibration_success=False, calibration_error="all models failed",
                         metrics={}, validation_success=False)
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == FAIL
    assert [c.check_id for c in result.checks] == ["calibration_success"]
    assert "all models failed" in result.reasons[0]


def test_missing_holdout_is_fail():
    evidence = _evidence(validation_success=False, validation_error="Train 캘리브레이션 실패")
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == FAIL
    assert _statuses(result)["holdout_available"] == "fail"


def test_train_test_leakage_is_fail():
    evidence = _evidence(train_frame_ids=["a", "b"], test_frame_ids=["b", "c"])
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == FAIL
    assert _statuses(result)["split_no_leakage"] == "fail"


def test_recommended_model_but_quality_below_standard_is_fail():
    """추천은 성공했지만(recommended_model 존재) 품질 기준 미달이면 FAIL - 추천과 판정은 별개."""
    evidence = _evidence(metrics=GOOD_METRICS | {"test_rms_px": 1.4, "test_p95_px": 2.5})
    result = evaluate_acceptance(evidence, _policy())
    assert result.recommended_model == "brown_conrady"
    assert result.verdict == FAIL


def test_sanity_error_is_fail_and_sanity_warning_is_warn_by_default():
    error = _evidence(sanity_issues=[{"code": "non_finite", "severity": "error", "message": "NaN"}])
    assert evaluate_acceptance(error, _policy()).verdict == FAIL
    warning = _evidence(sanity_issues=[{"code": "rms_high", "severity": "warning", "message": "high"}])
    assert evaluate_acceptance(warning, _policy()).verdict == WARN


def test_existing_diagnosis_and_evidence_gate_mapping_is_configurable():
    evidence = _evidence(
        diagnosis_patterns=[{"code": "poor_coverage", "severity": "error", "title": "t", "recommendation": "r"}],
        holdout_evidence_status="insufficient_evidence",
        holdout_evidence_reasons=["Test 프레임 3장"],
    )
    assert evaluate_acceptance(evidence, _policy()).verdict == WARN

    strict = _policy(existing_checks={"diagnosis_error": "fail"})
    assert evaluate_acceptance(evidence, strict).verdict == FAIL

    relaxed = _policy(existing_checks={"diagnosis_error": "info", "holdout_evidence_insufficient": "info"})
    assert evaluate_acceptance(evidence, relaxed).verdict == PASS


def test_existing_grade_and_confidence_signals_produce_warn():
    evidence = _evidence(overall_grade="poor", confidence_level="LOW", confidence_score=40.0)
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == WARN
    statuses = _statuses(result)
    assert statuses["overall_grade"] == "warn" and statuses["confidence_level"] == "warn"


def test_result_records_policy_and_is_not_vehicle_approval():
    result = evaluate_acceptance(_evidence(), _policy())
    data = result.to_dict()
    assert data["policy"]["thresholds"]["test_rms_px"] == {"max": 0.5, "required": True, "on_violation": "fail"}
    assert "차량 적용 승인이 아닙니다" in data["note"]


@pytest.mark.parametrize("section, fragment", [
    ({"thresholds": {"unknown_metric": {"max": 1}}}, "unknown_metric"),
    ({"thresholds": {"test_rms_px": {"max": -1}}}, "음수"),
    ({"thresholds": {"test_rms_px": {"max": "0.5"}}}, "숫자"),
    ({"thresholds": {"test_rms_px": {"max": 0.5, "required": "yes"}}}, "required"),
    ({"thresholds": {"test_rms_px": {"max": 0.5, "on_violation": "panic"}}}, "on_violation"),
    ({"existing_checks": {"sanity_error": "ignore"}}, "sanity_error"),
    ({"existing_checks": {"made_up": "warn"}}, "made_up"),
    ({"surprise": 1}, "surprise"),
])
def test_invalid_policy_is_rejected(section, fragment):
    with pytest.raises(AcceptancePolicyError, match=fragment):
        parse_policy(section)


# ---------------------------------------------------------------------------
# 판정 근거 분류 (계산 실패 vs 품질 저하 등)
# ---------------------------------------------------------------------------


def test_calculation_failure_is_classified_separately_from_quality():
    failed = evaluate_acceptance(_evidence(calibration_success=False, metrics={}), _policy())
    assert failed.fail_kinds == ["calculation_failure"]

    degraded = evaluate_acceptance(_evidence(metrics=GOOD_METRICS | {"test_rms_px": 0.9}), _policy())
    assert degraded.fail_kinds == ["quality"]
    assert degraded.fail_reasons and all("test_rms_px" in r for r in degraded.fail_reasons)


def test_invalid_result_leakage_and_missing_metric_kinds():
    invalid = evaluate_acceptance(
        _evidence(sanity_issues=[{"code": "fx_non_positive", "severity": "error", "message": "fx<=0"}]), _policy(),
    )
    assert invalid.fail_kinds == ["invalid_result"]
    leak = evaluate_acceptance(_evidence(train_frame_ids=["a"], test_frame_ids=["a"]), _policy())
    assert "data_leakage" in leak.fail_kinds
    missing = evaluate_acceptance(_evidence(metrics=GOOD_METRICS | {"edge_rms_px": None}), _policy())
    assert missing.fail_kinds == ["missing_metric"]


def test_warn_reasons_and_notes_are_separated():
    evidence = _evidence(
        holdout_evidence_status="insufficient_evidence",
        diagnosis_patterns=[{"code": "limited_coverage", "severity": "warning", "title": "t", "recommendation": "r"}],
    )
    result = evaluate_acceptance(evidence, _policy())
    assert result.verdict == WARN
    assert result.warn_kinds == ["evidence"]
    assert any("Hold-out" in r for r in result.warn_reasons)
    assert any("Diagnosis WARNING" in n for n in result.notes)  # diagnosis_warning 기본 = info
    assert not result.fail_reasons


def test_unconfigured_criteria_is_reported_as_warn_kind():
    result = evaluate_acceptance(_evidence(), _policy(limits={k: None for k in EXAMPLE_LIMITS}))
    assert result.warn_kinds == ["criteria_missing"]
    assert any("운영 기준" in r for r in result.warn_reasons)
