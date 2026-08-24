"""
quality/quality_score.py

Aggregates M2 (Geometry), M3 (Generalization), M4 (Stability) into
category scores and one Overall Quality score, per evaluation_metric_spec.md
section 17-18 (the "Geometry / Generalization / Stability / Sensitivity"
categorization, with Sensitivity deferred -- no Perturbation metric exists
yet in this codebase).

Each category score reuses quality.normalization's 0-100 curve, so a
category's score and its classification (GOOD/WARNING/BAD) are always
mutually consistent (see normalization.py's module docstring).

Weighting across categories is EQUAL by default (1/3 each) since there is
not yet a data-driven or product basis to weight one category over another
-- this mirrors the spec's still-open "Normalization" item. Weights are a
parameter, not a hardcoded constant, so this can be revisited once more
real-world evaluation runs exist to justify a different split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math

from evaluation.edge_alignment import EdgeAlignmentResult
from evaluation.holdout_consistency import HoldoutConsistencyResult
from evaluation.multiframe_consistency import MultiFrameConsistencyResult
from quality.normalization import score_m2, score_std, score_to_classification


DEFAULT_WEIGHTS = {
    "geometry": 1.0 / 3.0,
    "generalization": 1.0 / 3.0,
    "stability": 1.0 / 3.0,
}


@dataclass
class CategoryScore:
    name: str               # "geometry" | "generalization" | "stability"
    metric_name: str        # "M2" | "M3" | "M4"
    valid: bool              # False if the underlying metric FAILed
    score: float             # 0-100, NaN if not valid
    classification: str      # GOOD | WARNING | BAD | FAIL
    raw_value_px: float      # the underlying px value the score was derived from (mean_px or std_px)
    floor_px: float
    summary: dict = field(default_factory=dict)  # extra stats for report display
    warnings: list[str] = field(default_factory=list)


@dataclass
class QualityScoreResult:
    categories: list[CategoryScore]
    weights_used: dict[str, float]
    num_valid_categories: int
    overall_score: float       # NaN if zero valid categories
    overall_classification: str  # GOOD | WARNING | BAD | FAIL
    warnings: list[str] = field(default_factory=list)

    def category(self, name: str) -> Optional[CategoryScore]:
        for c in self.categories:
            if c.name == name:
                return c
        return None


def _geometry_category(m2: EdgeAlignmentResult) -> CategoryScore:
    valid = m2.classification != "FAIL"
    score = score_m2(m2.mean_px, m2.floor_px) if valid else float("nan")
    return CategoryScore(
        name="geometry", metric_name="M2", valid=valid,
        score=score, classification=score_to_classification(score) if valid else "FAIL",
        raw_value_px=m2.mean_px, floor_px=m2.floor_px,
        summary={
            "mean_px": m2.mean_px, "median_px": m2.median_px, "p95_px": m2.p95_px,
            "num_edge_points": m2.num_edge_points,
        },
        warnings=list(m2.warnings),
    )


def _generalization_category(m3: HoldoutConsistencyResult) -> CategoryScore:
    valid = m3.classification != "FAIL"
    score = score_std(m3.std_across_blocks_px, m3.floor_px) if valid else float("nan")
    return CategoryScore(
        name="generalization", metric_name="M3", valid=valid,
        score=score, classification=score_to_classification(score) if valid else "FAIL",
        raw_value_px=m3.std_across_blocks_px, floor_px=m3.floor_px,
        summary={
            "mean_across_blocks_px": m3.mean_across_blocks_px,
            "std_across_blocks_px": m3.std_across_blocks_px,
            "range_px": m3.range_px,
            "num_valid_blocks": m3.num_valid_blocks,
        },
        warnings=list(m3.warnings),
    )


def _stability_category(m4: MultiFrameConsistencyResult) -> CategoryScore:
    valid = m4.classification != "FAIL"
    score = score_std(m4.std_across_frames_px, m4.floor_px) if valid else float("nan")
    return CategoryScore(
        name="stability", metric_name="M4", valid=valid,
        score=score, classification=score_to_classification(score) if valid else "FAIL",
        raw_value_px=m4.std_across_frames_px, floor_px=m4.floor_px,
        summary={
            "mean_across_frames_px": m4.mean_across_frames_px,
            "std_across_frames_px": m4.std_across_frames_px,
            "p95_across_frames_px": m4.p95_across_frames_px,
            "max_across_frames_px": m4.max_across_frames_px,
            "num_outlier_frames": m4.num_outlier_frames,
        },
        warnings=list(m4.warnings),
    )


def compute_quality_score(
    m2_result: EdgeAlignmentResult,
    m3_result: HoldoutConsistencyResult,
    m4_result: MultiFrameConsistencyResult,
    weights: Optional[dict] = None,
) -> QualityScoreResult:
    """
    Combine Geometry (M2) / Generalization (M3) / Stability (M4) into
    per-category scores and one weighted Overall Quality score.

    If a category's underlying metric FAILed, that category is EXCLUDED
    from the weighted average and the remaining weights are renormalized
    to sum to 1 -- rather than forcing a 0, which would conflate "we
    couldn't measure this" with "this measured as terrible". The
    exclusion is always surfaced via a warning so the person doesn't
    mistake a partial score for a complete one.

    If ALL categories FAILed, overall_score is NaN and
    overall_classification is "FAIL".

    If SOME (but not all) categories FAILed, overall_classification is
    capped at "WARNING" even if the renormalized score of the surviving
    categories would round-trip to "GOOD" -- a partial result should never
    look as trustworthy as a complete one. overall_score itself is left
    un-clamped (it still reflects the renormalized weighted average of the
    valid categories) so the underlying number remains inspectable; only
    the classification label is capped.
    """
    weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    expected_keys = {"geometry", "generalization", "stability"}
    if set(weights.keys()) != expected_keys:
        raise ValueError(f"weights must have exactly keys {expected_keys}, got {set(weights.keys())}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"weights must be non-negative, got {weights}")
    if math.isclose(sum(weights.values()), 0.0):
        raise ValueError("weights must not all be zero")

    categories = [
        _geometry_category(m2_result),
        _generalization_category(m3_result),
        _stability_category(m4_result),
    ]

    warnings: list[str] = []
    for c in categories:
        if not c.valid:
            warnings.append(
                f"Category '{c.name}' ({c.metric_name}) FAILed and was excluded "
                f"from the Overall Quality score."
            )
        warnings.extend(f"[{c.name}] {w}" for w in c.warnings)

    valid_categories = [c for c in categories if c.valid]

    if not valid_categories:
        warnings.append("All categories FAILed; Overall Quality cannot be computed.")
        return QualityScoreResult(
            categories=categories,
            weights_used=weights,
            num_valid_categories=0,
            overall_score=float("nan"),
            overall_classification="FAIL",
            warnings=warnings,
        )

    valid_weight_sum = sum(weights[c.name] for c in valid_categories)
    if math.isclose(valid_weight_sum, 0.0):
        warnings.append(
            "All valid categories have zero weight; Overall Quality cannot be computed "
            "even though some categories are individually valid."
        )
        return QualityScoreResult(
            categories=categories,
            weights_used=weights,
            num_valid_categories=len(valid_categories),
            overall_score=float("nan"),
            overall_classification="FAIL",
            warnings=warnings,
        )

    renormalized = {c.name: weights[c.name] / valid_weight_sum for c in valid_categories}
    overall_score = sum(c.score * renormalized[c.name] for c in valid_categories)
    overall_classification = score_to_classification(overall_score)

    if len(valid_categories) < len(categories):
        warnings.append(
            f"Overall Quality is based on {len(valid_categories)}/{len(categories)} "
            f"categories (weights renormalized to {renormalized})."
        )
        # A category FAILing outright ("couldn't measure this") is strictly
        # worse information than a low-but-valid score ("measured, and it's
        # bad") -- so a partial Overall Quality must never read as better
        # than WARNING, no matter how good the surviving categories look.
        # Without this cap, e.g. 2/3 categories FAILing (M2 + M3 both unable
        # to run) while the remaining category (M4) scores perfectly would
        # still report "100.0 GOOD", which is misleading for a --fail-on-bad
        # CI gate: it would pass even though most of the calibration could
        # not actually be evaluated.
        if overall_classification == "GOOD":
            overall_classification = "WARNING"
            warnings.append(
                "Overall Quality classification capped at WARNING because not all "
                "categories could be measured -- a partial result cannot be reported "
                "as GOOD, even if the categories that were measured scored well."
            )

    return QualityScoreResult(
        categories=categories,
        weights_used=weights,
        num_valid_categories=len(valid_categories),
        overall_score=float(overall_score),
        overall_classification=overall_classification,
        warnings=warnings,
    )
