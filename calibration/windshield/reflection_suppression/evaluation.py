"""
calibration.windshield.reflection_suppression.evaluation
==============================================================

STEP 7 - Before/After evaluation using the SAME STEP 6 evaluator.

    Original  -> evaluate_reflection() -> Before Metrics
    Suppressed -> evaluate_reflection() -> After Metrics  (same reference, same config)

STEP 6의 `evaluate_reflection()`을 전혀 수정하지 않고 그대로 두 번 호출한다
(사용자 스펙 0/41번, "SAME Reflection Evaluator"). Reduction 계산은 이
모듈이 추가로 담당한다 - Evaluation package 자체를 건드리지 않는다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from calibration.windshield.reflection.evaluator import evaluate_reflection
from calibration.windshield.reflection.types import ReflectionEvaluationConfig
from calibration.windshield.reflection_suppression.types import ReflectionSuppressionEvaluation, ReflectionSuppressionResult

_EPS = 1e-6


def _reduction(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return float(1.0 - after / (before + _EPS))


def evaluate_suppression(
    normal_image: np.ndarray,
    suppression_result: ReflectionSuppressionResult,
    reference_image: Optional[np.ndarray] = None,
    config: Optional[ReflectionEvaluationConfig] = None,
    *,
    clean_roi_mask: Optional[np.ndarray] = None,
) -> ReflectionSuppressionEvaluation:
    """Before(원본)/After(suppressed) 각각에 STEP 6 evaluator를 그대로
    적용하고, reduction/retention 요약을 덧붙인다.

    `clean_roi_mask`(선택, HxW bool/0-1 array): reflection이 거의 없다고
    알려진 영역 - 주어지면 Over-suppression Metric(사용자 스펙 47번)도
    계산한다."""
    cfg = config or ReflectionEvaluationConfig(mode="reference" if reference_image is not None else "no_reference")
    before = evaluate_reflection(normal_image, reference_image, cfg, pair_id="before")

    if not suppression_result.success or suppression_result.suppressed_image is None:
        return ReflectionSuppressionEvaluation(
            before=before,
            after=before,
            success=False,
            error_message=suppression_result.error_message or "suppression did not produce an image to evaluate",
        )

    after = evaluate_reflection(suppression_result.suppressed_image, reference_image, cfg, pair_id="after")

    # 안정화 라운드 항목 2(CRITICAL) - STEP 6의 원칙("No-reference는 heuristic
    # likelihood일 뿐 ground-truth reflection measurement가 아니다")을 여기서
    # 다시 깨뜨리지 않는다. Reference Mode의 실제 reflection 측정값
    # (reflection_mean/p95/coverage)과 No-Reference Mode의 heuristic
    # likelihood(mean_strength, no-reference에서는 reflection_likelihood와
    # 같은 값)를 같은 "Reduction" 필드에 섞어 넣지 않는다 - 모드별로 완전히
    # 분리된 필드 집합을 채운다.
    reflection_mean_reduction = None
    reflection_p95_reduction = None
    coverage_reduction = None
    reflection_likelihood_before = None
    reflection_likelihood_after = None
    reflection_likelihood_reduction = None

    if before.mode == "reference" and after.mode == "reference":
        reflection_mean_reduction = _reduction(before.reflection_mean, after.reflection_mean)
        reflection_p95_reduction = _reduction(before.reflection_p95, after.reflection_p95)
        coverage_reduction = _reduction(before.reflection_coverage, after.reflection_coverage)
    else:
        reflection_likelihood_before = before.reflection_likelihood
        reflection_likelihood_after = after.reflection_likelihood
        reflection_likelihood_reduction = _reduction(before.reflection_likelihood, after.reflection_likelihood)

    over_suppression_score = None
    if clean_roi_mask is not None:
        mask = np.asarray(clean_roi_mask).astype(bool)
        if mask.shape == normal_image.shape[:2] and np.any(mask):
            diff = np.abs(
                suppression_result.suppressed_image.astype(np.float32) - normal_image.astype(np.float32)
            ) / 255.0
            over_suppression_score = float(np.mean(diff[mask]))

    return ReflectionSuppressionEvaluation(
        before=before,
        after=after,
        reflection_mean_reduction=reflection_mean_reduction,
        reflection_p95_reduction=reflection_p95_reduction,
        coverage_reduction=coverage_reduction,
        reflection_likelihood_before=reflection_likelihood_before,
        reflection_likelihood_after=reflection_likelihood_after,
        reflection_likelihood_reduction=reflection_likelihood_reduction,
        edge_retention_after=after.edge_retention,
        contrast_retention_after=after.contrast_retention,
        over_suppression_score=over_suppression_score,
        success=True,
    )
