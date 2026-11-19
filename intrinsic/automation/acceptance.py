"""Automatic PASS / WARN / FAIL acceptance of a finished intrinsic calibration.

모델 추천(``calibration.recommender``)과 합격 판정은 분리한다. 이 모듈은
추천 점수·가중치를 읽지도 바꾸지도 않고, 이미 계산된 결과(.ccproj로 저장된
CalibrationProject)에서 지표를 꺼내 세 종류의 검사로 판정만 한다.

1. Hard validity (고정 규칙, 설정 불가)
   계산 성공, Hold-out 성공, Train/Test 누수 없음. 하나라도 어긋나면 FAIL.
2. Existing signals (기존 코드가 이미 내리는 판단의 매핑)
   sanity_check / diagnosis / hold-out evidence gate / 종합 등급·신뢰도.
   기존 기준값은 그대로 두고, 그 결과를 fail|warn|info 중 무엇으로 볼지만
   프로파일에서 정한다.
3. Operational thresholds (운영 품질 기준)
   Test RMS, P95 등 사용자가 프로파일에 직접 적는 값. 값이 정해지지 않았으면
   (null) 임의로 채우지 않고 NOT_CONFIGURED로 남기며, 필수 지표의 기준이
   비어 있으면 PASS를 줄 수 없다(WARN + criteria_complete=false).

판정은 계산 결과에 대한 품질 평가일 뿐 차량 적용 승인이 아니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# 판정 정책에서 쓰는 개별 검사 상태
STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_INFO = "info"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_NOT_CONFIGURED = "not_configured"

_SEVERITY_TO_STATUS = {"fail": STATUS_FAIL, "warn": STATUS_WARN, "info": STATUS_INFO}


@dataclass(frozen=True)
class MetricSpec:
    direction: str  # "max": 값이 limit 이하여야 함, "min": 값이 limit 이상이어야 함
    description: str
    unit: str = ""


# 판정에 쓸 수 있는 지표. 이름에 단위를 넣어 프로파일만 보고도 의미가 드러나게 한다.
METRICS: dict[str, MetricSpec] = {
    "train_rms_px": MetricSpec("max", "Train RMS (선택 모델, 전체 프레임 학습 결과)", "px"),
    "test_rms_px": MetricSpec("max", "Hold-out Test RMS (Train 분할로만 학습)", "px"),
    "test_p95_px": MetricSpec("max", "Hold-out Test 코너 오차 P95", "px"),
    "test_p99_px": MetricSpec("max", "Hold-out Test 코너 오차 P99", "px"),
    "edge_rms_px": MetricSpec("max", "Hold-out Edge RMS", "px"),
    "train_test_gap_px": MetricSpec("max", "Hold-out Test RMS - Hold-out Train RMS", "px"),
    "stability_pct": MetricSpec("min", "All-parameter stability (bootstrap 우선, 없으면 covariance)", "%"),
    "observability_score": MetricSpec("min", "Jacobian 기반 observability score (0~100)"),
    "coverage_pct": MetricSpec("min", "4x4 이미지 영역 coverage", "%"),
    "num_images_used": MetricSpec("min", "계산에 사용된(활성) 프레임 수"),
    "num_test_frames": MetricSpec("min", "Hold-out 평가에 성공한 Test 프레임 수"),
}

# 기존 코드 신호 → 판정 상태 매핑의 기본값.
# sanity ERROR는 물리적으로 불가능한 파라미터(NaN, 초점거리 범위 밖 등)라
# FAIL로 본다. diagnosis ERROR는 "조치가 필요한 패턴"이지 수치 오류는 아니므로
# 사람이 검토하도록 WARN으로 둔다. 모두 프로파일에서 바꿀 수 있다.
DEFAULT_EXISTING_CHECKS: dict[str, str] = {
    "sanity_error": "fail",
    "sanity_warning": "warn",
    "diagnosis_error": "warn",
    "diagnosis_warning": "info",
    "holdout_evidence_insufficient": "warn",
    "grade_warning_or_poor": "warn",
    "confidence_low_or_reject": "warn",
}


class AcceptancePolicyError(ValueError):
    """프로파일의 acceptance 섹션이 잘못됨."""


@dataclass(frozen=True)
class ThresholdRule:
    metric: str
    limit: float | None
    required: bool
    on_violation: str  # "fail" | "warn"


@dataclass(frozen=True)
class AcceptancePolicy:
    thresholds: dict[str, ThresholdRule]
    existing_checks: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "thresholds": {
                name: {
                    METRICS[name].direction: rule.limit,
                    "required": rule.required,
                    "on_violation": rule.on_violation,
                }
                for name, rule in self.thresholds.items()
            },
            "existing_checks": dict(self.existing_checks),
        }


def parse_policy(section: Any) -> AcceptancePolicy:
    """프로파일 ``acceptance`` 섹션을 검증해 AcceptancePolicy로 만든다."""
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise AcceptancePolicyError("acceptance 섹션은 key: value 객체여야 합니다.")
    unknown = set(section) - {"thresholds", "existing_checks"}
    if unknown:
        raise AcceptancePolicyError(f"acceptance에 알 수 없는 키: {', '.join(sorted(unknown))}")

    problems: list[str] = []
    raw_thresholds = section.get("thresholds") or {}
    if not isinstance(raw_thresholds, dict):
        raise AcceptancePolicyError("acceptance.thresholds는 객체여야 합니다.")
    thresholds: dict[str, ThresholdRule] = {}
    for name, raw in raw_thresholds.items():
        if name not in METRICS:
            problems.append(
                f"acceptance.thresholds.{name}: 알 수 없는 지표입니다 "
                f"(사용 가능: {', '.join(METRICS)})."
            )
            continue
        spec = METRICS[name]
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            problems.append(f"acceptance.thresholds.{name}: {{{spec.direction}: 값, required: bool}} 형식이어야 합니다.")
            continue
        allowed = {spec.direction, "required", "on_violation"}
        extra = set(raw) - allowed
        if extra:
            wrong_direction = "min" if spec.direction == "max" else "max"
            hint = (
                f" ({name}는 '{spec.direction}' 기준 지표입니다)"
                if wrong_direction in extra else ""
            )
            problems.append(f"acceptance.thresholds.{name}: 알 수 없는 키 {sorted(extra)}{hint}.")
            continue
        limit = raw.get(spec.direction)
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
                problems.append(f"acceptance.thresholds.{name}.{spec.direction}: 유한한 숫자 또는 null이어야 합니다.")
                continue
            if limit < 0:
                problems.append(f"acceptance.thresholds.{name}.{spec.direction}: 음수일 수 없습니다.")
                continue
        required = raw.get("required", False)
        if not isinstance(required, bool):
            problems.append(f"acceptance.thresholds.{name}.required: true/false여야 합니다.")
            continue
        on_violation = raw.get("on_violation", "fail")
        if on_violation not in ("fail", "warn"):
            problems.append(f"acceptance.thresholds.{name}.on_violation: fail 또는 warn이어야 합니다.")
            continue
        thresholds[name] = ThresholdRule(
            metric=name,
            limit=None if limit is None else float(limit),
            required=required,
            on_violation=on_violation,
        )

    raw_existing = section.get("existing_checks") or {}
    if not isinstance(raw_existing, dict):
        raise AcceptancePolicyError("acceptance.existing_checks는 객체여야 합니다.")
    existing = dict(DEFAULT_EXISTING_CHECKS)
    for key, value in raw_existing.items():
        if key not in DEFAULT_EXISTING_CHECKS:
            problems.append(
                f"acceptance.existing_checks.{key}: 알 수 없는 항목입니다 "
                f"(사용 가능: {', '.join(DEFAULT_EXISTING_CHECKS)})."
            )
        elif value not in _SEVERITY_TO_STATUS:
            problems.append(f"acceptance.existing_checks.{key}: fail, warn, info 중 하나여야 합니다.")
        else:
            existing[key] = value

    if problems:
        raise AcceptancePolicyError("\n".join(problems))
    return AcceptancePolicy(thresholds=thresholds, existing_checks=existing)


# ---------------------------------------------------------------------------
# Evidence: 계산 결과에서 판정에 필요한 값만 추린 평평한 구조
# ---------------------------------------------------------------------------


@dataclass
class AcceptanceEvidence:
    chosen_model: str | None
    recommended_model: str | None
    calibration_success: bool
    calibration_error: str | None = None
    validation_success: bool = False
    validation_error: str | None = None
    train_frame_ids: list[str] = field(default_factory=list)
    test_frame_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)
    sanity_issues: list[dict] = field(default_factory=list)  # {code, severity, message}
    diagnosis_patterns: list[dict] = field(default_factory=list)  # {code, severity, title, recommendation}
    holdout_evidence_status: str | None = None
    holdout_evidence_reasons: list[str] = field(default_factory=list)
    overall_grade: str | None = None
    confidence_level: str | None = None
    confidence_score: float | None = None


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def collect_evidence(project) -> AcceptanceEvidence:
    """CalibrationProject(.ccproj에서 불러온 것)에서 판정 근거를 추린다.

    값이 없으면 None으로 남긴다 - 절대 기본값으로 채우지 않는다.
    """
    from calibration.models.common import infer_image_size
    from calibration.quality import coverage_percentage
    from calibration.sanity_check import run_sanity_check

    final = project.final_result
    scores = project.model_scores or []
    recommended = next((s.model_name for s in scores if s.is_recommended), None)
    chosen = final.chosen_model if final is not None else recommended

    evidence = AcceptanceEvidence(
        chosen_model=getattr(chosen, "value", chosen),
        recommended_model=getattr(recommended, "value", recommended),
        calibration_success=False,
    )
    if chosen is None:
        evidence.calibration_error = "선택(추천)된 모델이 없습니다."
        return evidence

    cal = project.calibration_results.get(chosen)
    val = project.validation_results.get(chosen)
    if cal is None:
        evidence.calibration_error = f"{evidence.chosen_model} 계산 결과가 프로젝트에 없습니다."
        return evidence
    evidence.calibration_success = bool(cal.success)
    evidence.calibration_error = cal.error_message

    metrics: dict[str, float | None] = {name: None for name in METRICS}
    metrics["train_rms_px"] = _finite_or_none(cal.rms_error)
    if val is not None:
        evidence.validation_success = bool(val.success) and _finite_or_none(val.test_rms) is not None
        evidence.validation_error = val.error_message
        evidence.train_frame_ids = list(val.train_frame_ids)
        evidence.test_frame_ids = list(val.test_frame_ids)
        if val.success:
            metrics["test_rms_px"] = _finite_or_none(val.test_rms)
            metrics["edge_rms_px"] = _finite_or_none(val.edge_rms)
            if val.test_residual_stats is not None:
                metrics["test_p95_px"] = _finite_or_none(val.test_residual_stats.p95)
                metrics["test_p99_px"] = _finite_or_none(val.test_residual_stats.p99)
            train = _finite_or_none(val.train_rms)
            test = metrics["test_rms_px"]
            if train is not None and test is not None:
                metrics["train_test_gap_px"] = test - train
            failed = set(val.failed_test_frame_ids)
            metrics["num_test_frames"] = float(len([i for i in val.test_frame_ids if i not in failed]))
        if val.evidence_gate is not None:
            evidence.holdout_evidence_status = val.evidence_gate.status
            evidence.holdout_evidence_reasons = list(val.evidence_gate.reasons)

    # diagnosis._diagnose_bootstrap_stability와 같은 출처 우선순위
    uncertainty = cal.param_uncertainty_bootstrap or cal.param_uncertainty
    if uncertainty is not None:
        metrics["stability_pct"] = _finite_or_none(uncertainty.overall_stability)
    if cal.observability is not None:
        metrics["observability_score"] = _finite_or_none(cal.observability.observability_score)

    coverage = final.dataset_coverage_pct if final is not None else None
    if coverage is None and project.dataset.coverage_grid:
        coverage = coverage_percentage(project.dataset.coverage_grid)
    metrics["coverage_pct"] = _finite_or_none(coverage)
    metrics["num_images_used"] = float(project.dataset.num_enabled)
    evidence.metrics = metrics

    if cal.success:
        image_size = infer_image_size(project.dataset, project.camera_config)
        sanity = run_sanity_check(cal, project.camera_config, image_size)
        evidence.sanity_issues = [
            {"code": i.code, "severity": getattr(i.severity, "value", str(i.severity)), "message": i.message}
            for i in sanity.issues
        ]

    if final is not None:
        evidence.overall_grade = getattr(final.overall_grade, "value", final.overall_grade)
        if final.confidence is not None:
            evidence.confidence_level = final.confidence.level
            evidence.confidence_score = _finite_or_none(final.confidence.score)
        if final.diagnosis is not None:
            evidence.diagnosis_patterns = [
                {
                    "code": p.code,
                    "severity": getattr(p.severity, "value", str(p.severity)),
                    "title": p.title,
                    "recommendation": p.recommendation,
                }
                for p in final.diagnosis.patterns
            ]
    return evidence


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    category: str  # "validity" | "existing" | "threshold"
    status: str
    message: str
    metric: str | None = None
    value: float | None = None
    limit: float | None = None
    direction: str | None = None
    required: bool | None = None
    # 판정 근거의 종류 (FAIL/WARN이 "왜" 나왔는지 구분하기 위함)
    #   calculation_failure : 계산 자체가 실패 (모델 계산 / Hold-out 학습·평가)
    #   invalid_result      : 계산은 끝났지만 값이 물리적으로 무효 (sanity ERROR)
    #   data_leakage        : Train/Test 분할 누수
    #   missing_metric      : 판정에 필요한 지표가 계산되지 않음
    #   quality             : 품질 저하 (임계값 미달, sanity WARNING, 기존 등급/신뢰도)
    #   diagnosis           : 기존 diagnosis 패턴 (촬영 보완 권고 포함)
    #   evidence            : Hold-out 근거 부족
    #   criteria_missing    : 운영 기준 미정
    kind: str = ""


@dataclass
class AcceptanceResult:
    verdict: str
    criteria_complete: bool
    reasons: list[str]
    checks: list[CheckResult]
    chosen_model: str | None
    recommended_model: str | None
    policy: dict
    fail_reasons: list[str] = field(default_factory=list)
    warn_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    fail_kinds: list[str] = field(default_factory=list)
    warn_kinds: list[str] = field(default_factory=list)
    note: str = "합격 판정은 계산 결과의 품질 평가이며 차량 적용 승인이 아닙니다."

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.4g}{unit}"


def _validity_checks(evidence: AcceptanceEvidence) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if evidence.calibration_success:
        checks.append(CheckResult("calibration_success", "validity", STATUS_PASS,
                                  f"{evidence.chosen_model} 계산 성공"))
    else:
        checks.append(CheckResult(
            "calibration_success", "validity", STATUS_FAIL,
            f"계산 실패: {evidence.calibration_error or '선택 모델 결과 없음'}",
        ))
        return checks

    if evidence.validation_success:
        checks.append(CheckResult("holdout_available", "validity", STATUS_PASS, "Hold-out 검증 성공"))
    else:
        checks.append(CheckResult(
            "holdout_available", "validity", STATUS_FAIL,
            f"Hold-out 검증 결과가 없거나 실패했습니다: {evidence.validation_error or 'Test RMS 없음'}",
        ))

    overlap = sorted(set(evidence.train_frame_ids) & set(evidence.test_frame_ids))
    if overlap:
        checks.append(CheckResult(
            "split_no_leakage", "validity", STATUS_FAIL,
            f"Train/Test 분할에 겹치는 프레임이 있습니다: {overlap[:5]}",
        ))
    elif evidence.test_frame_ids:
        checks.append(CheckResult(
            "split_no_leakage", "validity", STATUS_PASS,
            f"Train {len(evidence.train_frame_ids)} / Test {len(evidence.test_frame_ids)} 프레임, 중복 없음",
        ))
    return checks


def _existing_checks(evidence: AcceptanceEvidence, mapping: dict[str, str]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def add(check_id: str, key: str, message: str, kind: str) -> None:
        checks.append(CheckResult(check_id, "existing", _SEVERITY_TO_STATUS[mapping[key]], message, kind=kind))

    for issue in evidence.sanity_issues:
        error = issue["severity"] == "error"
        add(
            f"sanity:{issue['code']}", "sanity_error" if error else "sanity_warning",
            f"Sanity check {issue['severity'].upper()}: {issue['message']}",
            "invalid_result" if error else "quality",
        )
    for pattern in evidence.diagnosis_patterns:
        if pattern["severity"] == "error":
            key = "diagnosis_error"
        elif pattern["severity"] == "warning":
            key = "diagnosis_warning"
        else:
            continue
        add(
            f"diagnosis:{pattern['code']}", key,
            f"Diagnosis {pattern['severity'].upper()}: {pattern['title']} → {pattern['recommendation']}",
            "diagnosis",
        )
    if evidence.holdout_evidence_status == "insufficient_evidence":
        reasons = "; ".join(evidence.holdout_evidence_reasons) or "근거 부족"
        add("holdout_evidence", "holdout_evidence_insufficient", f"Hold-out 근거 부족: {reasons}", "evidence")
    if evidence.overall_grade in ("warning", "poor"):
        add("overall_grade", "grade_warning_or_poor", f"기존 종합 등급이 {evidence.overall_grade.upper()}입니다.", "quality")
    if evidence.confidence_level in ("LOW", "REJECT"):
        add(
            "confidence_level", "confidence_low_or_reject",
            f"기존 신뢰도 수준이 {evidence.confidence_level}입니다 (score {_fmt(evidence.confidence_score)}).",
            "quality",
        )
    return checks


def _threshold_checks(evidence: AcceptanceEvidence, policy: AcceptancePolicy) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for name, rule in policy.thresholds.items():
        spec = METRICS[name]
        value = _finite_or_none(evidence.metrics.get(name))
        base = dict(metric=name, value=value, limit=rule.limit, direction=spec.direction, required=rule.required)
        if value is None:
            if rule.required:
                checks.append(CheckResult(
                    f"threshold:{name}", "threshold", STATUS_FAIL,
                    f"필수 지표 {name}({spec.description})가 계산되지 않았습니다.", **base,
                ))
            else:
                checks.append(CheckResult(
                    f"threshold:{name}", "threshold", STATUS_NOT_AVAILABLE,
                    f"선택 지표 {name}가 계산되지 않아 판정에서 제외했습니다.", **base,
                ))
            continue
        if rule.limit is None:
            checks.append(CheckResult(
                f"threshold:{name}", "threshold", STATUS_NOT_CONFIGURED,
                f"{name} = {_fmt(value, spec.unit)}: 운영 기준이 정해지지 않았습니다 (profile에서 null).",
                **base,
            ))
            continue
        ok = value <= rule.limit if spec.direction == "max" else value >= rule.limit
        op = "≤" if spec.direction == "max" else "≥"
        if ok:
            checks.append(CheckResult(
                f"threshold:{name}", "threshold", STATUS_PASS,
                f"{name} = {_fmt(value, spec.unit)} {op} {_fmt(rule.limit, spec.unit)}", **base,
            ))
        else:
            checks.append(CheckResult(
                f"threshold:{name}", "threshold", _SEVERITY_TO_STATUS[rule.on_violation],
                f"{name} = {_fmt(value, spec.unit)} 기준 미달 (요구: {op} {_fmt(rule.limit, spec.unit)})",
                **base,
            ))
    return checks


_VALIDITY_KINDS = {
    "calibration_success": "calculation_failure",
    "holdout_available": "calculation_failure",
    "split_no_leakage": "data_leakage",
}


def _assign_kind(check: CheckResult) -> None:
    if check.kind:
        return
    if check.category == "validity":
        check.kind = _VALIDITY_KINDS.get(check.check_id, "invalid_result")
    elif check.category == "threshold":
        if check.status == STATUS_NOT_CONFIGURED:
            check.kind = "criteria_missing"
        elif check.value is None:
            check.kind = "missing_metric"
        else:
            check.kind = "quality"


def evaluate_acceptance(evidence: AcceptanceEvidence, policy: AcceptancePolicy) -> AcceptanceResult:
    checks = _validity_checks(evidence)
    if evidence.calibration_success:
        checks += _existing_checks(evidence, policy.existing_checks)
        checks += _threshold_checks(evidence, policy)
    for check in checks:
        _assign_kind(check)

    required_unconfigured = [
        c.metric for c in checks
        if c.category == "threshold" and c.status == STATUS_NOT_CONFIGURED and c.required
    ]
    required_missing = any(c.category == "threshold" and c.required and c.value is None for c in checks)
    criteria_complete = evidence.calibration_success and not required_missing and not required_unconfigured and any(
        r.required for r in policy.thresholds.values()
    )

    failures = [c for c in checks if c.status == STATUS_FAIL]
    warnings = [c for c in checks if c.status == STATUS_WARN]
    reasons: list[str] = [c.message for c in failures] + [c.message for c in warnings]
    if failures:
        verdict = FAIL
    elif warnings or not criteria_complete:
        verdict = WARN
        if required_unconfigured:
            reasons.append(
                "필수 지표의 운영 기준이 정해지지 않아 PASS를 줄 수 없습니다: "
                + ", ".join(required_unconfigured)
            )
        elif not any(r.required for r in policy.thresholds.values()):
            reasons.append("필수(required) 품질 기준이 하나도 지정되지 않아 PASS를 줄 수 없습니다.")
    else:
        verdict = PASS
        reasons.append("모든 필수 기준과 기존 유효성 검사를 통과했습니다.")

    notes = [c.message for c in checks if c.status in (STATUS_INFO, STATUS_NOT_AVAILABLE, STATUS_NOT_CONFIGURED)]
    return AcceptanceResult(
        verdict=verdict,
        criteria_complete=criteria_complete,
        reasons=reasons,
        checks=checks,
        chosen_model=evidence.chosen_model,
        recommended_model=evidence.recommended_model,
        policy=policy.to_dict(),
        fail_reasons=[c.message for c in failures],
        warn_reasons=[c.message for c in warnings] + (
            [r for r in reasons if r.startswith("필수")] if verdict == WARN else []
        ),
        notes=notes,
        fail_kinds=sorted({c.kind for c in failures}),
        warn_kinds=sorted({c.kind for c in warnings} | ({"criteria_missing"} if verdict == WARN and not criteria_complete else set())),
    )
