"""
camera_calibrator.calibration.recommender
=============================================

설계 문서 8번 - 모델 자동 추천 로직.

    "RMS가 가장 낮은 모델 = 정답은 절대 금지."
    Score = w1*E_train + w2*E_test + w3*E_edge + w4*E_line + w5*P

이 모듈은 의도적으로 validation_results(Hold-out, 5단계)를 필수 인자로
요구한다 - Train RMS만 갖고 추천하는 걸 API 차원에서 막기 위해서다.
straightness(E_line)는 V2 기능이라 아직 없으므로, 있으면 쓰고 없으면
그 항목은 점수에 영향을 주지 않게(중립 0점) 처리한다.

추천은 "정답 지정"이 아니라 "근거 제시"다 (설계 문서 8번 "추천과 선택의 분리").
그래서 이 모듈의 결과는 항상 ModelScore 리스트 + 사람이 읽을 수 있는 근거
문장이며, 최종 모델을 실제로 쓸지는 여전히 호출하는 쪽(UI/사용자)의 몫이다.
"""

from __future__ import annotations

import math

from calibration.types import (
    CalibrationResult,
    CalibrationConfidenceReport,
    CalibrationMethod,
    CameraModelType,
    DiagnosisSeverity,
    FinalResult,
    ModelScore,
    ModelScoreWeights,
    OutlierResult,
    QualityGrade,
    ValidationResult,
)
from calibration.diagnosis import diagnose_calibration
from calibration.error_normalization import reference_equivalent_error

# 모델별 distortion 자유도. 실제 CalibrationResult.distortion
# 배열 길이로 역산하지 않고 여기서 명시하는 이유: Pinhole은 distortion 배열이
# [0,0,0,0,0]으로 채워져 있어도 "고정된 0"이지 "추정된 0"이 아니므로 자유도는 0이다.
_DISTORTION_FREE_PARAMS = {
    CameraModelType.PINHOLE: 0,
    CameraModelType.BROWN_CONRADY: 5,
    CameraModelType.EXTENDED_PINHOLE: 8,    # Rational, 항상 8계수 (k1~k6,p1,p2)
    CameraModelType.FISHEYE: 4,             # k1~k4
}

# 모델 선택용 정보 기준(AIC/BIC)에 쓰는 명목 파라미터 개수.
# 모든 모델이 공통으로 추정하는 intrinsic 4개(fx, fy, cx, cy)에 모델별
# distortion 자유도를 더한다. 프레임별 extrinsic은 모델 간 같은 데이터에서
# 거의 공통 비용이라 여기서는 제외한다 - AIC/BIC를 "카메라 모델 복잡도" 비교에
# 쓰기 위한 정의다.
_INTRINSIC_FREE_PARAMS = 4


def _complexity_counts() -> dict[CameraModelType, int]:
    return dict(_DISTORTION_FREE_PARAMS)


def parameter_count_for_model(model: CameraModelType) -> int:
    """AIC/BIC에 사용할 모델별 자유 파라미터 수.

    Pinhole은 distortion을 전부 0으로 고정하지만 intrinsic 4개는 추정하므로
    k=4다. Extended(Rational, 항상 8계수)/Fisheye는 여기에 각 distortion
    자유도를 더한다.
    """
    return _INTRINSIC_FREE_PARAMS + _complexity_counts().get(model, 0)


def compute_information_criteria(
    *,
    residual_sum_squares: float | None,
    num_observations: int,
    parameter_count: int,
) -> tuple[float | None, float | None]:
    """AIC=2k+n*ln(RSS/n), BIC=k*ln(n)+n*ln(RSS/n).

    RSS가 0이면 수학적으로 log(0)이 되어 -inf가 되지만, 실제 subpixel 데이터에서
    완전 0은 숫자/합성 특이 케이스에 가깝다. 비교표에 무한대를 흘리지 않도록
    machine epsilon으로 바닥을 둔다.
    """
    if residual_sum_squares is None or num_observations <= 0 or parameter_count < 0:
        return None, None
    rss = max(float(residual_sum_squares), float(num_observations) * 1e-12)
    n = float(num_observations)
    k = float(parameter_count)
    log_likelihood_term = n * math.log(rss / n)
    return 2.0 * k + log_likelihood_term, k * math.log(n) + log_likelihood_term


def _rss_from_result(result: CalibrationResult) -> tuple[float | None, int]:
    """CalibrationResult의 코너 포인트 단위 residual stats에서 RSS와 n을 복원."""
    stats = result.residual_stats
    if not result.success or stats is None or stats.n <= 0 or stats.rmse is None:
        return None, 0
    return float(stats.rmse ** 2 * stats.n), int(stats.n)


def _relative_gap(best: float | None, runner_up: float | None) -> float | None:
    """낮을수록 좋은 두 metric 사이의 상대 차이.

    예: 0.42 vs 0.43이면 약 2.4% 차이. 두 값이 너무 작거나 없으면 None.
    """
    if best is None or runner_up is None:
        return None
    denom = max(abs(best), 1e-12)
    return abs(runner_up - best) / denom


def _test_p95(result: ValidationResult | None) -> float | None:
    stats = result.test_residual_stats if result else None
    return stats.p95 if stats else None


def _model_p95(calibration: CalibrationResult, validation: ValidationResult | None) -> float | None:
    return _test_p95(validation)


def model_selection_status(
    calibration: CalibrationResult | None,
    validation: ValidationResult | None,
) -> tuple[bool, str, str | None]:
    if calibration is None or not calibration.success:
        reason = "Calibration failed."
        if calibration and calibration.error_message:
            reason = f"Calibration failed: {calibration.error_message}"
        return False, "CALIBRATION FAILED", reason
    if validation is None:
        return False, "VALIDATION INCOMPLETE", "Hold-out validation was not run."
    if not validation.success:
        return False, "VALIDATION INCOMPLETE", validation.error_message or "Hold-out validation failed."
    if not validation.test_frame_ids:
        return False, "VALIDATION INCOMPLETE", validation.error_message or "No hold-out test frames."
    if validation.test_rms is None:
        return False, "VALIDATION INCOMPLETE", "Hold-out Test RMS is unavailable."
    if validation.test_residual_stats is None:
        return False, "VALIDATION INCOMPLETE", "Hold-out test residual statistics are unavailable."
    return True, "ELIGIBLE", None


def is_model_selection_eligible(
    calibration: CalibrationResult | None,
    validation: ValidationResult | None,
) -> bool:
    eligible, _status, _reason = model_selection_status(calibration, validation)
    return eligible


def _radial_error_score(result: CalibrationResult) -> float | None:
    """Radial bands에서 외곽 쪽 P95/RMS를 대표 radial penalty로 뽑는다."""
    profile = result.radial_bands or result.radial_profile
    if not result.success or profile is None or not profile.bins:
        return None
    edge_labels = {"outer", "edge", "corner"}
    values: list[float] = []
    for b in profile.bins:
        label = (b.label or "").lower()
        if label and label not in edge_labels:
            continue
        value = b.p95_error if b.p95_error is not None else b.rms_error
        if value is None:
            value = b.mean_error
        if value is not None:
            values.append(float(value))
    if not values:
        for b in profile.bins:
            value = b.p95_error if b.p95_error is not None else b.rms_error
            if value is None:
                value = b.mean_error
            if value is not None:
                values.append(float(value))
    return max(values) if values else None


def _parameter_stability_penalty(result: CalibrationResult) -> float | None:
    """Parameter stability는 높을수록 좋으므로 100-stability penalty로 변환."""
    pu = result.param_uncertainty_bootstrap or result.param_uncertainty
    if not result.success or pu is None:
        return None
    values: list[float] = []
    if pu.overall_stability is not None:
        values.append(float(pu.overall_stability))
    for v in (pu.fx_stability, pu.fy_stability, pu.cx_stability, pu.cy_stability):
        if v is not None:
            values.append(float(v))
    for d in pu.distortion_stats:
        if d.stability_score is not None:
            values.append(float(d.stability_score))
    if not values:
        return None
    stability = max(0.0, min(100.0, sum(values) / len(values)))
    return 100.0 - stability


def _observability_penalty(result: CalibrationResult) -> float | None:
    """Rank/condition/correlation을 0(좋음)~100(나쁨) penalty로 압축."""
    obs = result.observability
    if not result.success or obs is None or obs.jacobian_cols <= 0:
        return None
    penalties: list[float] = []
    if obs.rank < obs.jacobian_cols:
        penalties.append((obs.jacobian_cols - obs.rank) / obs.jacobian_cols * 100.0)
    if obs.condition_number is not None:
        if math.isinf(obs.condition_number):
            penalties.append(100.0)
        elif obs.condition_number > 0:
            # 1e4 이하는 양호, 1e12 이상은 매우 불량으로 보는 로그 스케일.
            penalties.append(max(0.0, min(100.0, (math.log10(obs.condition_number) - 4.0) / 8.0 * 100.0)))
    if obs.max_abs_correlation is not None:
        # 0.8 이하 상관은 거의 무시하고 0.99 이상은 강한 coupling으로 본다.
        penalties.append(max(0.0, min(100.0, (obs.max_abs_correlation - 0.8) / 0.19 * 100.0)))
    return max(penalties) if penalties else 0.0


def _observability_positive_evidence_allowed(result: CalibrationResult) -> bool:
    obs = result.observability
    return bool(
        obs is not None
        and obs.observability_score is not None
        and obs.observability_score >= 50.0
        and obs.observability_grade != "POOR"
    )


def _parameter_stability_score(result: CalibrationResult) -> float | None:
    penalty = _parameter_stability_penalty(result)
    return 100.0 - penalty if penalty is not None else None


def _is_best_lower(values: dict[CameraModelType, float | None], model: CameraModelType) -> bool:
    value = values.get(model)
    valid = [v for v in values.values() if v is not None]
    return value is not None and valid and value <= min(valid) + 1e-9


def _is_best_higher(values: dict[CameraModelType, float | None], model: CameraModelType) -> bool:
    value = values.get(model)
    valid = [v for v in values.values() if v is not None]
    return value is not None and valid and value >= max(valid) - 1e-9


def _all_present(values: dict[CameraModelType, float | None], models: list[CameraModelType]) -> bool:
    return bool(models) and all(values.get(m) is not None for m in models)


def _normalize_comparable(
    values: dict[CameraModelType, float | None],
    comparable_models: list[CameraModelType],
) -> dict[CameraModelType, float]:
    if not comparable_models:
        return {k: 0.0 for k in values}
    if any(values.get(m) is None for m in comparable_models):
        return {k: 0.0 for k in values}
    masked = {k: (v if k in comparable_models else None) for k, v in values.items()}
    return _normalize(masked)


def _fmt_px(value: float) -> str:
    return f"{value:.3f}px"


def _fmt_plain(value: float) -> str:
    return f"{value:.3f}"


def _apply_selection_reasons(
    scores: list[ModelScore],
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
) -> None:
    recommended = next((s for s in scores if s.is_recommended), None)
    if recommended is None:
        return

    model = recommended.model_name
    score_by_model = {s.model_name: s for s in scores}
    models = [
        s.model_name for s in scores
        if is_model_selection_eligible(calibration_results.get(s.model_name), validation_results.get(s.model_name))
    ]
    test_rms = {
        m: validation_results[m].test_rms
        for m in models
    }
    test_p95 = {m: _model_p95(calibration_results[m], validation_results.get(m)) for m in models}
    edge = {m: validation_results[m].edge_rms for m in models}
    line = {
        m: validation_results[m].straightness_residual
        if validation_results[m].straightness_source == "test" else None
        for m in models
    }
    radial = {m: _radial_error_score(calibration_results[m]) for m in models}
    aic = {m: score_by_model[m].aic for m in models}
    bic = {m: score_by_model[m].bic for m in models}
    stability = {m: _parameter_stability_score(calibration_results[m]) for m in models}
    observability = {m: _observability_penalty(calibration_results[m]) for m in models}

    reasons: list[str] = []

    lower_metrics = [
        ("Lowest Test RMS", test_rms, _fmt_px, True),
        ("Lowest Test P95", test_p95, _fmt_px, _all_present(test_p95, models)),
        ("Lowest Edge Error", edge, _fmt_px, _all_present(edge, models)),
        ("Best Test Straightness", line, _fmt_px, _all_present(line, models)),
        ("Best Radial Error", radial, _fmt_px, _all_present(radial, models)),
        ("Best AIC", aic, _fmt_plain, _all_present(aic, models)),
        ("Best BIC", bic, _fmt_plain, _all_present(bic, models)),
    ]
    for label, values, formatter, comparable in lower_metrics:
        if not comparable:
            continue
        value = values.get(model)
        if value is not None and _is_best_lower(values, model):
            reasons.append(f"{label} ({formatter(value)})")

    observability_value = observability.get(model)
    if (
        observability_value is not None
        and _all_present(observability, models)
        and _observability_positive_evidence_allowed(calibration_results[model])
        and _is_best_lower(observability, model)
    ):
        reasons.append(f"Best Observability (penalty {observability_value:.1f}/100)")

    stability_value = stability.get(model)
    if stability_value is not None:
        if _is_best_higher(stability, model):
            reasons.append(f"Most Stable Parameters ({stability_value:.1f}/100)")
        elif stability_value >= 90.0:
            reasons.append(f"Stable Parameters ({stability_value:.1f}/100)")

    if not reasons:
        evidence_keys = set(recommended.evidence_components)
        strongest = sorted(
            ((k, v) for k, v in recommended.components.items() if k in evidence_keys),
            key=lambda kv: kv[1],
        )[:3]
        labels = {
            "train": "Train RMS",
            "test": "Test RMS",
            "edge": "Edge RMS",
            "line": "Line Straightness",
            "complexity": "Complexity",
            "p95": "Test P95",
            "radial": "Radial Error",
            "aic": "AIC",
            "bic": "BIC",
            "stability": "Parameter Stability",
            "observability": "Observability",
        }
        reasons = [f"Low weighted {labels.get(k, k)} penalty ({v:.3f})" for k, v in strongest]

    recommended.selection_reasons = reasons[:8]


def _confidence_level(confidence: float) -> str:
    if confidence >= 80.0:
        return "HIGH"
    if confidence >= 50.0:
        return "MEDIUM"
    return "LOW"


def _apply_per_model_confidence(scores: list[ModelScore], ranked: list[ModelScore]) -> None:
    """모든 모델에 confidence를 채운다.

    점수는 낮을수록 좋으므로 best score와의 차이를 softmax 확률처럼 바꾼다.
    절대적인 통계 확률은 아니지만, 문서가 요구하는 "Pinhole/Extended/Fisheye
    각각 confidence"를 같은 스케일에서 비교하기 위한 상대 지지율이다.
    """
    failed = [s for s in scores if s not in ranked]
    for s in failed:
        s.selection_confidence = 0.0
        s.selection_confidence_level = "LOW"
        s.selection_confidence_reason = (
            s.selection_ineligibility_reason
            or "This model is not a selection candidate."
        )

    if not ranked:
        return
    if len(ranked) == 1:
        only = ranked[0]
        only.selection_confidence = 100.0
        only.selection_confidence_level = "HIGH"
        only.selection_confidence_reason = "Only one model has complete hold-out validation evidence."
        return

    best_score = ranked[0].score
    temperature = 0.12
    supports = [math.exp(-(s.score - best_score) / temperature) for s in ranked]
    total_support = sum(supports)
    for rank, (score, support) in enumerate(zip(ranked, supports), start=1):
        confidence = 100.0 * support / total_support if total_support > 0 else 0.0
        score.selection_confidence = round(confidence, 1)
        score.selection_confidence_level = _confidence_level(score.selection_confidence)
        score.selection_confidence_reason = (
            f"Rank {rank}/{len(ranked)} by weighted model score; "
            f"score={score.score:.3f}, confidence derived from score margin."
        )


def _apply_selection_confidence(
    scores: list[ModelScore],
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    complexity_counts: dict[CameraModelType, int],
) -> None:
    """추천 모델과 2위 모델의 차이를 기반으로 confidence를 채운다.

    기준은 보수적으로 잡는다. score 차이가 작거나, 실제 원본 metric(test RMS,
    test P95, edge RMS)이 거의 같으면 LOW로 낮춘다. 즉 "추천은 이 모델이지만
    둘이 사실상 비슷하다"는 상황을 숨기지 않는다.
    """
    eligible = [s for s in scores if s.is_selection_eligible]
    ranked = sorted(eligible, key=lambda s: (round(s.score, 6), complexity_counts.get(s.model_name, 0)))
    _apply_per_model_confidence(scores, ranked)
    if not ranked:
        return

    best = ranked[0]
    if len(ranked) == 1:
        return

    runner = ranked[1]
    score_gap = runner.score - best.score
    best_val = validation_results.get(best.model_name)
    runner_val = validation_results.get(runner.model_name)
    metric_gaps = {
        "Test RMS": _relative_gap(best_val.test_rms if best_val else None, runner_val.test_rms if runner_val else None),
        "Test P95": _relative_gap(_test_p95(best_val), _test_p95(runner_val)),
        "Edge RMS": _relative_gap(best_val.edge_rms if best_val else None, runner_val.edge_rms if runner_val else None),
    }
    close_metrics = [name for name, gap in metric_gaps.items() if gap is not None and gap <= 0.03]

    if score_gap <= 0.05 or close_metrics:
        level = "LOW"
        confidence = max(0.0, min(100.0, 35.0 + score_gap / 0.05 * 15.0))
        reason = (
            f"{_LABELS[best.model_name]} and {_LABELS[runner.model_name]} perform similarly"
            f" (score gap {score_gap:.3f}"
            + (f", close metrics: {', '.join(close_metrics)}" if close_metrics else "")
            + ")."
        )
    elif score_gap <= 0.12:
        level = "MEDIUM"
        confidence = max(50.0, min(79.0, 50.0 + (score_gap - 0.05) / 0.07 * 29.0))
        reason = (
            f"{_LABELS[best.model_name]} leads {_LABELS[runner.model_name]}, "
            f"but the margin is moderate (score gap {score_gap:.3f})."
        )
    else:
        level = "HIGH"
        confidence = max(80.0, min(99.0, 80.0 + min(score_gap, 0.30) / 0.30 * 19.0))
        reason = (
            f"{_LABELS[best.model_name]} is clearly separated from {_LABELS[runner.model_name]} "
            f"(score gap {score_gap:.3f})."
        )

    # 추천 모델에는 "각 모델별 상대 confidence"에 더해, 1위와 2위가 충분히
    # 갈렸는지 보는 selection confidence를 보수적으로 반영한다.
    best.selection_confidence = round(min(best.selection_confidence or confidence, confidence), 1)
    best.selection_confidence_level = level
    best.selection_confidence_reason = reason


def _normalize(values: dict[CameraModelType, float | None]) -> dict[CameraModelType, float]:
    """0(가장 좋음)~1(가장 나쁨)로 min-max 정규화.

    - 값이 전부 None(예: 프레임이 너무 적어 어떤 모델도 straightness를 계산
      못한 경우)이면 전부 0.0을 줘서 그 항목이 점수에 영향을 주지 않게 한다.
    - 일부만 None(그 모델만 실패했거나 데이터가 없음)이면 그 모델에는 1.0(최악)을
      줘서 불이익을 준다 - "몰라서 좋아 보이는" 상황을 방지.
    """
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return {k: 0.0 for k in values}

    lo, hi = min(valid.values()), max(valid.values())
    rng = hi - lo
    out: dict[CameraModelType, float] = {}
    for k, v in values.items():
        if v is None:
            out[k] = 1.0
        else:
            out[k] = (v - lo) / rng if rng > 0 else 0.0
    return out


def compute_model_scores(
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    weights: ModelScoreWeights = ModelScoreWeights(),
) -> list[ModelScore]:
    """Standard 4모델(Ideal Pinhole/Brown-Conrady/Rational/Fisheye)의 Score를
    계산하고, 가장 낮은(=좋은) 모델에 is_recommended=True 표시.

    Object-Releasing(Advanced)은 calibration_results에 섞여 들어와도 여기서
    걸러진다(calibration_method != OBJECT_RELEASING인 것만 남김) - AIC/BIC의
    파라미터 개수 가정이 "카메라 파라미터만 최적화" 전제인데, Object-Releasing은
    타겟 형상까지 함께 최적화하는 추가 자유도가 있어 같은 기준으로 비교할 수
    없기 때문이다. Object-Releasing 자체 품질은
    calibration/object_releasing_validation.py의 전용 Hold-out/비교로 따로 본다.

    Args:
        calibration_results: 모델별 (Outlier pruning까지 마친) 최종 CalibrationResult.
            E_train, E_edge(fallback)의 출처.
        validation_results: 모델별 ValidationResult (5단계 Hold-out 결과).
            E_test, E_edge(우선 출처)의 출처. 반드시 "같은 train/test 분할"로
            계산된 것이어야 공정하다 (validate_all_models 참고).
    """
    models = [
        m for m, result in calibration_results.items()
        if result.calibration_method != CalibrationMethod.OBJECT_RELEASING
    ]
    complexity_counts = _complexity_counts()
    eligibility = {
        m: model_selection_status(calibration_results.get(m), validation_results.get(m))
        for m in models
    }
    eligible_models = [m for m in models if eligibility[m][0]]

    e_train = {m: (calibration_results[m].rms_error if eligibility[m][0] else None) for m in models}

    e_test = {}
    for m in models:
        vr = validation_results.get(m)
        e_test[m] = vr.test_rms if eligibility[m][0] else None

    # Edge는 Hold-out(test)에서 나온 값만 쓴다. Train regional_error는 Test Edge가 아니다.
    e_edge = {}
    for m in models:
        vr = validation_results.get(m)
        e_edge[m] = vr.edge_rms if eligibility[m][0] and vr else None

    # Straightness (calibration/straightness.py, validate_holdout에서 채워짐).
    # Model selection에서는 hold-out test frame에서 측정된 값만 사용한다.
    # Train fallback은 UI/diagnostic 표시용이며 selection evidence가 아니다.
    e_line = {}
    for m in models:
        vr = validation_results.get(m)
        e_line[m] = (
            vr.straightness_residual
            if eligibility[m][0] and vr and vr.straightness_source == "test"
            else None
        )

    complexity = {m: float(complexity_counts.get(m, 0)) if eligibility[m][0] else None for m in models}

    n_train = _normalize(e_train)
    n_test = _normalize(e_test)
    n_edge = _normalize_comparable(e_edge, eligible_models)
    n_line = _normalize_comparable(e_line, eligible_models)
    n_complexity = _normalize(complexity)

    raw_aic: dict[CameraModelType, float | None] = {}
    raw_bic: dict[CameraModelType, float | None] = {}
    raw_p95 = {
        m: _model_p95(calibration_results[m], validation_results.get(m))
        if eligibility[m][0] else None
        for m in models
    }
    raw_radial = {m: _radial_error_score(calibration_results[m]) if eligibility[m][0] else None for m in models}
    raw_stability = {
        m: _parameter_stability_penalty(calibration_results[m]) if eligibility[m][0] else None
        for m in models
    }
    raw_observability = {
        m: _observability_penalty(calibration_results[m]) if eligibility[m][0] else None
        for m in models
    }

    scores: list[ModelScore] = []
    for m in models:
        parameter_count = parameter_count_for_model(m)
        rss, n_obs = _rss_from_result(calibration_results[m])
        aic, bic = compute_information_criteria(
            residual_sum_squares=rss,
            num_observations=n_obs,
            parameter_count=parameter_count,
        )
        raw_aic[m] = aic
        raw_bic[m] = bic

    n_p95 = _normalize_comparable(raw_p95, eligible_models)
    n_radial = _normalize_comparable(raw_radial, eligible_models)
    n_aic = _normalize_comparable(raw_aic, eligible_models)
    n_bic = _normalize_comparable(raw_bic, eligible_models)
    n_stability = _normalize_comparable(raw_stability, eligible_models)
    n_observability = _normalize_comparable(raw_observability, eligible_models)
    comparable_evidence = {
        "train": bool(eligible_models),
        "test": bool(eligible_models),
        "edge": _all_present(e_edge, eligible_models),
        "line": _all_present(e_line, eligible_models),
        "complexity": bool(eligible_models),
        "p95": _all_present(raw_p95, eligible_models),
        "radial": _all_present(raw_radial, eligible_models),
        "aic": _all_present(raw_aic, eligible_models),
        "bic": _all_present(raw_bic, eligible_models),
        "stability": _all_present(raw_stability, eligible_models),
        "observability": _all_present(raw_observability, eligible_models),
    }

    for m in models:
        parameter_count = parameter_count_for_model(m)
        rss, n_obs = _rss_from_result(calibration_results[m])
        aic, bic = compute_information_criteria(
            residual_sum_squares=rss,
            num_observations=n_obs,
            parameter_count=parameter_count,
        )
        is_eligible, status, ineligibility_reason = eligibility[m]
        components = {}
        total = math.inf
        if is_eligible:
            components = {
                "train": weights.w_train * n_train[m],
                "test": weights.w_test * n_test[m],
                "edge": weights.w_edge * n_edge[m],
                "line": weights.w_line * n_line[m],
                "complexity": weights.w_complexity * n_complexity[m],
                "p95": weights.w_p95 * n_p95[m],
                "radial": weights.w_radial * n_radial[m],
                "aic": weights.w_aic * n_aic[m],
                "bic": weights.w_bic * n_bic[m],
                "stability": weights.w_stability * n_stability[m],
                "observability": weights.w_observability * n_observability[m],
            }
            total = sum(components.values())
        evidence_components = [
            key for key in components
            if comparable_evidence.get(key, False)
            and (key != "observability" or _observability_positive_evidence_allowed(calibration_results[m]))
        ]
        scores.append(
            ModelScore(
                model_name=m,
                score=total,
                components=components,
                evidence_components=evidence_components,
                is_selection_eligible=is_eligible,
                selection_status=status,
                selection_ineligibility_reason=ineligibility_reason,
                parameter_count=parameter_count,
                residual_sum_squares=rss,
                num_observations=n_obs,
                aic=aic,
                bic=bic,
            )
        )

    extended_score = next((s for s in scores if s.model_name == CameraModelType.EXTENDED_PINHOLE), None)
    fisheye_score = next((s for s in scores if s.model_name == CameraModelType.FISHEYE), None)
    if (
        extended_score is not None
        and fisheye_score is not None
        and extended_score.is_selection_eligible
        and fisheye_score.is_selection_eligible
        and extended_score.score < fisheye_score.score
    ):
        extended_p95 = raw_p95.get(CameraModelType.EXTENDED_PINHOLE)
        fisheye_p95 = raw_p95.get(CameraModelType.FISHEYE)
        fisheye_test = e_test.get(CameraModelType.FISHEYE)
        extended_test = e_test.get(CameraModelType.EXTENDED_PINHOLE)
        score_gap = fisheye_score.score - extended_score.score
        validation_tied = (
            score_gap <= max(weights.w_stability, 0.05)
            and fisheye_test is not None
            and extended_test is not None
            and fisheye_test <= extended_test + 0.03
            and (
                extended_p95 is None
                or fisheye_p95 is None
                or fisheye_p95 <= extended_p95 + 0.03
            )
        )
        if validation_tied:
            adjustment = -(score_gap + 1e-6)
            fisheye_score.components["fisheye_validation_tie_break"] = adjustment
            fisheye_score.score += adjustment

    # 완성된 hold-out core evidence가 있는 모델만 추천 후보로 인정한다.
    eligible = [s for s in scores if s.is_selection_eligible]
    if eligible:
        # 점수가 같으면 더 단순한 모델을 우선한다 (Occam's razor).
        best = min(eligible, key=lambda s: (round(s.score, 6), complexity_counts.get(s.model_name, 0)))
        best.is_recommended = True
        _apply_selection_reasons(scores, calibration_results, validation_results)
        _apply_selection_confidence(scores, calibration_results, validation_results, complexity_counts)

    return scores


# ---------------------------------------------------------------------------
# 근거 문장 생성 (설계 문서 8번 "근거 있는 추천")
# ---------------------------------------------------------------------------

_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}


def _frame_count_fairness_warning(
    calibration_results: dict[CameraModelType, CalibrationResult],
) -> str | None:
    """모델별로 실제 학습(fit)에 쓰인 프레임 수(used_frame_count)가 서로 다르면
    Train RMS를 모델 간 직접 비교하는 게 공정하지 않다는 경고를 만든다 -
    Fisheye의 robust fallback(불안정한 프레임 자동 제외)이 전형적인 원인이다.
    Hold-out Test RMS는 모든 모델이 동일한 test_ids에서 K/D를 고정한 채
    solvePnP만으로 평가되므로(설계 문서 9번) 이 불공정성의 영향을 받지 않는다 -
    이 경고는 Train RMS 비교에만 해당한다.
    """
    counts = {
        m: cal.used_frame_count
        for m, cal in calibration_results.items()
        if cal and cal.success and cal.used_frame_count > 0
    }
    if len(set(counts.values())) <= 1:
        return None
    parts = [f"{_LABELS.get(m, m.value)} {n}장" for m, n in counts.items()]
    return (
        "⚠ 모델별 Train 프레임 수가 다릅니다 (" + ", ".join(parts) + ") - "
        "일부 모델이 불안정한 프레임을 자동 제외했을 수 있습니다(Fisheye 흔함). "
        "Train RMS는 모델 간 직접 비교하지 마세요 - Test/Hold-out RMS는 모든 모델이 "
        "동일한 프레임에서 평가되므로 영향받지 않습니다."
    )


def build_recommendation_message(
    scores: list[ModelScore],
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
) -> str:
    """"⭐ Recommended: Extended Pinhole - Test RMS/Edge RMS 기준 가장 우수"
    형태의 사람이 읽는 근거 문장. 어디까지나 근거 제시이지 강제가 아니다.
    """
    recommended = next((s for s in scores if s.is_recommended), None)
    if recommended is None:
        calibrated = [m for m, cal in calibration_results.items() if cal and cal.success]
        if calibrated:
            statuses = []
            for m in calibrated:
                _eligible, status, reason = model_selection_status(
                    calibration_results.get(m), validation_results.get(m)
                )
                statuses.append(
                    f"{_LABELS.get(m, m.value)}: {status}" + (f" - {reason}" if reason else "")
                )
            return (
                "캘리브레이션은 성공했지만 완전한 Hold-out validation evidence를 가진 모델이 없어 "
                "자동 추천을 수행하지 않았습니다. Validation Status와 Failure Reason을 확인하세요."
                + (f"\n{chr(10).join(statuses)}" if statuses else "")
            )
        return "모든 모델의 캘리브레이션이 실패해 추천할 수 없습니다."

    model = recommended.model_name
    label = _LABELS[model]
    train_rms = calibration_results[model].rms_error
    vr = validation_results.get(model)
    test_rms = vr.test_rms if vr else None

    # "line"(직선성)처럼 프레임 부족 등으로 아무 모델도 값을 갖지 못한 경우엔
    # 전부 0으로 정규화되어 "가장 유리한 항목"처럼 보이지만, 실제로는 측정된
    # 적이 없다. 측정이 하나라도 있었던 지표만 근거 후보로 인정한다.
    line_measured = any(
        (
            is_model_selection_eligible(calibration_results.get(m), validation_results.get(m))
            and validation_results[m].straightness_source == "test"
            and validation_results[m].straightness_residual is not None
        )
        for m in validation_results
    )
    metric_names_kr = {
        "train": "Train RMS",
        "test": "Test RMS",
        "edge": "Edge RMS",
        "line": "직선성",
        "complexity": "복잡도",
        "p95": "Test P95",
        "radial": "Radial Error",
        "aic": "AIC",
        "bic": "BIC",
        "stability": "Parameter Stability",
        "observability": "Observability",
    }
    if line_measured:
        metric_names_kr["line"] = "직선성"
    else:
        metric_names_kr.pop("line", None)

    reasons = []
    # 가장 기여도가 낮은(=가장 유리했던) 항목을 근거로 든다
    evidence_keys = set(recommended.evidence_components)
    sorted_components = sorted(
        (
            (k, v) for k, v in recommended.components.items()
            if k in metric_names_kr and k in evidence_keys
        ),
        key=lambda kv: kv[1],
    )
    top_reasons = [metric_names_kr[k] for k, v in sorted_components[:2] if v <= 0.15]
    if top_reasons:
        reasons.append(" / ".join(top_reasons) + " 기준에서 가장 우수")

    detail = []
    if train_rms is not None:
        detail.append(f"Train {train_rms:.3f}px")
    if test_rms is not None:
        detail.append(f"Test {test_rms:.3f}px")

    msg = f"⭐ 추천 모델: {label} (Score={recommended.score:.3f})"
    if recommended.selection_confidence_level:
        msg += (
            f" | Confidence={recommended.selection_confidence_level}"
            + (f" {recommended.selection_confidence:.0f}%" if recommended.selection_confidence is not None else "")
        )
    if detail:
        msg += f" [{', '.join(detail)}]"
    if recommended.selection_reasons:
        msg += "\nReasons:"
        for reason in recommended.selection_reasons:
            msg += f"\n✓ {reason}"
    elif reasons:
        msg += f"\nReasons:\n✓ {', '.join(reasons)}"
    if recommended.selection_confidence_level == "LOW" and recommended.selection_confidence_reason:
        msg += f"\n⚠ Model selection confidence: LOW - {recommended.selection_confidence_reason}"
    frame_count_warning = _frame_count_fairness_warning(calibration_results)
    if frame_count_warning:
        msg += f"\n{frame_count_warning}"
    msg += "\n※ 이 추천은 근거 제시일 뿐입니다. ROS 파이프라인 호환성 등의 이유로 다른 모델을 선택해도 됩니다."
    return msg


def format_score_table(
    scores: list[ModelScore],
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
) -> str:
    """설계 문서 8번 모델 비교 예시 형식.

                     Pinhole   Extended   Fisheye
    Train RMS          0.91      0.42       0.38
    Test RMS           1.03      0.49       0.44
    Edge RMS           1.72      0.61       0.48
    Complexity           ★        ★★       ★★★
    Score              0.62      0.18       0.31
    Recommendation                 ⭐
    """
    order = [s.model_name for s in scores]
    labels = [_LABELS[m] for m in order]
    col_w = max(10, max(len(l) for l in labels) + 2)

    def row(name: str, values: list[str]) -> str:
        return f"{name:<14}" + "".join(f"{v:>{col_w}}" for v in values)

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    score_by_model = {s.model_name: s for s in scores}

    train_vals = [fmt(calibration_results[m].rms_error) if calibration_results[m].success else "FAIL" for m in order]
    test_vals = [fmt(validation_results.get(m).test_rms if validation_results.get(m) else None) for m in order]
    p95_vals = [fmt(_model_p95(calibration_results[m], validation_results.get(m))) for m in order]
    # edge는 components(가중치 적용된 정규화 값)가 아니라 원본 px 단위로 다시 뽑아서 보여준다.
    raw_edge = []
    for m in order:
        vr = validation_results.get(m)
        if vr and vr.success and vr.edge_rms is not None:
            raw_edge.append(fmt(vr.edge_rms))
        else:
            raw_edge.append("N/A")

    score_vals = [
        f"{score_by_model[m].score:.3f}"
        if score_by_model.get(m) and score_by_model[m].is_selection_eligible
        else "Not eligible"
        for m in order
    ]
    radial_vals = [fmt(_radial_error_score(calibration_results[m])) for m in order]
    aic_vals = [fmt(score_by_model[m].aic) if score_by_model.get(m) else "N/A" for m in order]
    bic_vals = [fmt(score_by_model[m].bic) if score_by_model.get(m) else "N/A" for m in order]
    stability_vals = [
        fmt(100.0 - _parameter_stability_penalty(calibration_results[m]))
        if _parameter_stability_penalty(calibration_results[m]) is not None else "N/A"
        for m in order
    ]
    observability_vals = [
        fmt(_observability_penalty(calibration_results[m]),) if _observability_penalty(calibration_results[m]) is not None else "N/A"
        for m in order
    ]
    confidence_vals = [
        (
            f"{score_by_model[m].selection_confidence_level} "
            f"{score_by_model[m].selection_confidence:.0f}%"
            if score_by_model.get(m)
            and score_by_model[m].selection_confidence_level
            and score_by_model[m].selection_confidence is not None
            else ""
        )
        for m in order
    ]
    rec_vals = ["⭐" if score_by_model.get(m) and score_by_model[m].is_recommended else "" for m in order]

    selection_status_vals = [
        score_by_model[m].selection_status if score_by_model.get(m) else "N/A"
        for m in order
    ]

    lines = [
        row("", labels),
        row("Train RMS", train_vals),
        row("Test RMS", test_vals),
        row("Test P95", p95_vals),
        row("Edge RMS", raw_edge),
        row("Radial", radial_vals),
        row("AIC", aic_vals),
        row("BIC", bic_vals),
        row("Stability", stability_vals),
        row("Obs Penalty", observability_vals),
        row("Selection", selection_status_vals),
        row("Score", score_vals),
        row("Confidence", confidence_vals),
        row("Recommend", rec_vals),
    ]
    return "\n".join(lines)


def compare_model_rankings(
    scores_before: list[ModelScore],
    scores_after: list[ModelScore],
) -> str:
    """설계 문서 17번 - "Outlier 제거 전후 효과 측정"의 마지막 항목,
    "model ranking 변화". outlier 제거 전/후 각각 compute_model_scores()로
    나온 두 ModelScore 리스트를 비교해, 추천 모델이 바뀌었는지와 각 모델의
    순위/점수가 어떻게 움직였는지 보여준다.

    주의: 이 프로젝트의 Score는 "낮을수록 좋다"(오차 성격의 가중합, docstring
    compute_model_scores 참고) - 그래서 오름차순(reverse=False)으로 정렬해야
    1위가 실제로 가장 좋은(점수가 가장 낮은) 모델이 된다.

        Model Ranking: Before -> After
        1위  Fisheye (0.612)  ->  Extended Pinhole (0.598)
        2위  Extended Pinhole (0.588)  ->  Fisheye (0.571)
        3위  Pinhole (0.301)  ->  Pinhole (0.295)

        ⚠ 추천 모델이 바뀌었습니다: Fisheye -> Extended Pinhole
    """
    if not scores_before or not scores_after:
        return "비교할 순위 정보가 없습니다."

    ranked_before = sorted((s for s in scores_before if s.is_selection_eligible), key=lambda s: s.score)
    ranked_after = sorted((s for s in scores_after if s.is_selection_eligible), key=lambda s: s.score)
    ineligible_before = [s for s in scores_before if not s.is_selection_eligible]
    ineligible_after = [s for s in scores_after if not s.is_selection_eligible]

    if not ranked_before and not ranked_after:
        return "비교할 selection-eligible 순위 정보가 없습니다."

    rec_before = next((s.model_name for s in scores_before if s.is_recommended), None)
    rec_after = next((s.model_name for s in scores_after if s.is_recommended), None)

    lines = ["Model Ranking: Before -> After"]
    n = max(len(ranked_before), len(ranked_after))
    for i in range(n):
        before_str = (
            f"{_LABELS[ranked_before[i].model_name]} ({ranked_before[i].score:.3f})"
            if i < len(ranked_before) else "N/A"
        )
        after_str = (
            f"{_LABELS[ranked_after[i].model_name]} ({ranked_after[i].score:.3f})"
            if i < len(ranked_after) else "N/A"
        )
        lines.append(f"{i+1}위  {before_str:<28} -> {after_str}")

    if ineligible_before or ineligible_after:
        lines.append("")
        lines.append("Not eligible:")
        for label, items in (("Before", ineligible_before), ("After", ineligible_after)):
            if not items:
                continue
            text = ", ".join(
                f"{_LABELS[s.model_name]} - {s.selection_status}"
                + (f" ({s.selection_ineligibility_reason})" if s.selection_ineligibility_reason else "")
                for s in items
            )
            lines.append(f"{label}: {text}")

    lines.append("")
    if rec_before is not None and rec_after is not None and rec_before != rec_after:
        lines.append(f"\u26a0 추천 모델이 바뀌었습니다: {_LABELS[rec_before]} -> {_LABELS[rec_after]}")
    else:
        lines.append("추천 모델은 바뀌지 않았습니다.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 최종 결과 조립 (설계 문서 12번, 18번 - FinalResult)
# ---------------------------------------------------------------------------

# 설계 문서 3.1번 RMS 등급표를 "종합 등급" 산정에도 재사용한다. 문서가 반복
# 경고하듯 이건 절대적 pass/fail 기준이 아니라 안내용 가이드라인이다 - 최종
# 판단은 여전히 사용자의 몫이며, 이 등급은 리포트에 "참고 지표"로만 나간다.
def _grade_for_value(v: float | None, camera_matrix=None) -> QualityGrade | None:
    if v is None:
        return None
    normalized_value = reference_equivalent_error(v, camera_matrix)
    if normalized_value is not None:
        v = normalized_value
    if v < 0.3:
        return QualityGrade.EXCELLENT
    if v < 0.5:
        return QualityGrade.VERY_GOOD
    if v < 1.0:
        return QualityGrade.GOOD
    if v < 2.0:
        return QualityGrade.WARNING
    return QualityGrade.POOR


_GRADE_BADNESS_ORDER = [
    QualityGrade.EXCELLENT,
    QualityGrade.VERY_GOOD,
    QualityGrade.GOOD,
    QualityGrade.WARNING,
    QualityGrade.POOR,
    QualityGrade.REJECT,
]


def _worst_grade(grades: list[QualityGrade]) -> QualityGrade:
    """여러 지표의 등급 중 "가장 나쁜" 등급을 최종 등급으로 채택한다.
    한 지표라도 나쁘면 종합 등급도 낮아져야 리포트가 낙관 편향되지 않는다
    (예: Train RMS는 훌륭한데 Edge RMS가 나쁘면 종합은 Edge 기준을 따라야 함).
    """
    if not grades:
        return QualityGrade.WARNING  # 판단 근거 자체가 없으면 "주의" 취급 (낙관 금지)
    return max(grades, key=_GRADE_BADNESS_ORDER.index)


def _score_lower_better(value: float | None, excellent: float, poor: float) -> float | None:
    if value is None:
        return None
    if value <= excellent:
        return 100.0
    if value >= poor:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (poor - value) / (poor - excellent)))


def _final_confidence_level(score: float, grade: QualityGrade) -> str:
    if grade == QualityGrade.REJECT or score < 50.0:
        return "REJECT"
    if score >= 85.0:
        return "HIGH"
    if score >= 70.0:
        return "MEDIUM"
    return "LOW"


def _cap_by_grade(score: float, grade: QualityGrade) -> float:
    caps = {
        QualityGrade.EXCELLENT: 100.0,
        QualityGrade.VERY_GOOD: 92.0,
        QualityGrade.GOOD: 84.0,
        QualityGrade.WARNING: 69.0,
        QualityGrade.POOR: 49.0,
        QualityGrade.REJECT: 0.0,
    }
    return min(score, caps.get(grade, 69.0))


def compute_final_confidence(
    cal: CalibrationResult | None,
    val: ValidationResult | None,
    overall_grade: QualityGrade,
    dataset_coverage_pct: float | None = None,
    scores: list[ModelScore] | None = None,
    diagnosis=None,
) -> CalibrationConfidenceReport:
    if cal is None or not cal.success:
        return CalibrationConfidenceReport(
            score=0.0,
            level="REJECT",
            reasons=["Calibration did not succeed."],
            warnings=[cal.error_message if cal and cal.error_message else "No usable calibration result."],
        )

    components: dict[str, float] = {}
    weights: dict[str, float] = {}

    def add(name: str, score: float | None, weight: float) -> None:
        if score is None:
            return
        components[name] = round(max(0.0, min(100.0, score)), 1)
        weights[name] = weight

    def quality_value(value: float | None) -> float | None:
        normalized = reference_equivalent_error(value, cal.camera_matrix)
        return value if normalized is None else normalized

    add("train_rms", _score_lower_better(quality_value(cal.rms_error), 0.25, 2.0), 0.14)
    if val and val.success:
        add("test_rms", _score_lower_better(quality_value(val.test_rms), 0.30, 2.0), 0.18)
        p95 = val.test_residual_stats.p95 if val.test_residual_stats else None
        add("test_p95", _score_lower_better(quality_value(p95), 0.60, 3.0), 0.14)
        add("edge_rms", _score_lower_better(quality_value(val.edge_rms), 0.40, 2.5), 0.12)
        add("straightness", _score_lower_better(quality_value(val.straightness_residual), 0.20, 1.5), 0.08)
    add("coverage", dataset_coverage_pct, 0.10)

    stability = _parameter_stability_score(cal)
    add("stability", stability, 0.08)
    if cal.observability and cal.observability.observability_score is not None:
        add("observability", cal.observability.observability_score, 0.08)
    if cal.undistortion_quality:
        add("undistortion", cal.undistortion_quality.quality_score, 0.06)

    chosen_score = next((s for s in scores or [] if s.model_name == cal.model_name), None)
    if chosen_score and chosen_score.selection_confidence is not None:
        add("model_selection", chosen_score.selection_confidence, 0.02)

    if components:
        total_weight = sum(weights.values())
        raw_score = sum(components[k] * weights[k] for k in components) / max(total_weight, 1e-12)
    else:
        raw_score = 0.0

    warnings: list[str] = []
    if not val or not val.success:
        raw_score -= 10.0
        warnings.append("Hold-out validation is missing or failed.")
    if dataset_coverage_pct is None:
        raw_score -= 5.0
        warnings.append("Dataset coverage is unavailable.")

    if diagnosis:
        n_errors = sum(1 for p in diagnosis.patterns if p.severity == DiagnosisSeverity.ERROR)
        n_warnings = sum(1 for p in diagnosis.patterns if p.severity == DiagnosisSeverity.WARNING)
        penalty = min(30.0, n_errors * 15.0 + n_warnings * 6.0)
        raw_score -= penalty
        if penalty > 0:
            warnings.append(f"Diagnosis patterns reduced confidence by {penalty:.0f} points.")

    score = round(_cap_by_grade(max(0.0, min(100.0, raw_score)), overall_grade), 1)
    strongest = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:3]
    weakest = sorted(components.items(), key=lambda kv: kv[1])[:3]
    reasons = [f"Strong {name}: {value:.1f}/100" for name, value in strongest]
    reasons += [f"Weak {name}: {value:.1f}/100" for name, value in weakest if value < 75.0]

    return CalibrationConfidenceReport(
        score=score,
        level=_final_confidence_level(score, overall_grade),
        components=components,
        reasons=reasons[:8],
        warnings=warnings,
    )


def compute_final_result(
    chosen_model: CameraModelType,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    dataset_coverage_pct: float | None = None,
    outlier_result: OutlierResult | None = None,
    corner_outlier_result: "CornerOutlierResult | None" = None,
    scores: list[ModelScore] | None = None,
    coverage_grid: "list[CoverageCell] | None" = None,
    dataset_diversity: "DiversityScores | None" = None,
) -> FinalResult:
    """사용자가 최종적으로 선택한 모델을 기준으로 FinalResult를 조립한다.

    "추천"(is_recommended)이 아니라 "선택"(chosen_model)을 기준으로 한다는
    점이 중요하다 - 설계 문서 8번 "추천과 선택의 분리" 원칙: 사용자가 추천과
    다른 모델을 골랐어도 리포트는 항상 실제 선택된 모델 기준으로 나가야 한다.
    """
    cal = calibration_results.get(chosen_model)
    val = validation_results.get(chosen_model)

    if cal is None or not cal.success:
        overall_grade = QualityGrade.REJECT
    else:
        candidate_values = [cal.rms_error]
        if val and val.success:
            candidate_values += [val.test_rms, val.edge_rms, val.straightness_residual]
        grades = [
            g for g in (_grade_for_value(v, cal.camera_matrix) for v in candidate_values)
            if g is not None
        ]
        overall_grade = _worst_grade(grades)

    diagnosis = (
        diagnose_calibration(cal, val, dataset_coverage_pct, coverage_grid, dataset_diversity)
        if cal is not None else None
    )
    confidence = compute_final_confidence(cal, val, overall_grade, dataset_coverage_pct, scores, diagnosis)

    return FinalResult(
        chosen_model=chosen_model,
        calibration=cal,
        validation=val,
        outlier=outlier_result,
        corner_outlier=corner_outlier_result,
        dataset_coverage_pct=dataset_coverage_pct,
        overall_grade=overall_grade,
        confidence=confidence,
        model_scores=scores or [],
        diagnosis=diagnosis,
    )
