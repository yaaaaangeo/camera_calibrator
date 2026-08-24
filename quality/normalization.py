"""
quality/normalization.py

Maps a raw metric value (px, for M2/M3/M4) into a 0-100 quality score,
anchored to the SAME sensor-relative floor(Z) multiplier thresholds already
used for GOOD/WARNING/BAD classification (quality.noise_floor). This keeps
"score" and "classification" mutually consistent by construction, rather
than being two independently-tuned systems that could disagree.

--------------------------------------------------------------------------
Design
--------------------------------------------------------------------------
Let r = value_px / floor_px (the same ratio classify() uses).

We want a smooth, monotonically decreasing curve score(r) such that:
  r = 0            -> score = 100   (perfect: no measurable error above floor)
  r = good_mult     -> score = 80    (GOOD/WARNING boundary)
  r = warning_mult  -> score = 50    (WARNING/BAD boundary, natural midpoint)
  r -> infinity     -> score -> 0

We use a generalized logistic / Hill-type curve:

    score(r) = 100 / (1 + (r / warning_mult)^p)

Pinning the r=warning_mult anchor at 50 forces r0 = warning_mult exactly
(since (warning_mult/warning_mult)^p = 1 for any p, giving 100/(1+1)=50).
The exponent p is then solved from the r=good_mult anchor:

    100 / (1 + (good_mult/warning_mult)^p) = 80
    => (good_mult/warning_mult)^p = 0.25
    => p = ln(0.25) / ln(good_mult/warning_mult)

This is solved once per multiplier pair (M2's 2x/5x vs M3/M4's STD 1x/3x)
and cached.

--------------------------------------------------------------------------
Why this matters: score bands align with GOOD/WARNING/BAD automatically
--------------------------------------------------------------------------
Because the curve is monotonically decreasing and passes exactly through
(good_mult, 80) and (warning_mult, 50):
  - every r < good_mult (GOOD)     maps to score > 80
  - every good_mult <= r < warning_mult (WARNING) maps to 50 <= score < 80
  - every r >= warning_mult (BAD)  maps to score < 50
So a person reading "score 92" and a person reading "classification GOOD"
are never in conflict -- score_to_classification() below implements the
same 80/50 boundary and is used both for per-metric and overall scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from quality.noise_floor import (
    M2_GOOD_MULTIPLIER,
    M2_WARNING_MULTIPLIER,
    STD_GOOD_MULTIPLIER,
    STD_WARNING_MULTIPLIER,
)


GOOD_WARNING_BOUNDARY_SCORE = 80.0
WARNING_BAD_BOUNDARY_SCORE = 50.0


@lru_cache(maxsize=None)
def _solve_exponent(good_mult: float, warning_mult: float, boundary_score: float = GOOD_WARNING_BOUNDARY_SCORE) -> float:
    """
    Solve p such that score(good_mult) == boundary_score, given the curve
    is pinned to pass through (warning_mult, 50) by construction (see
    module docstring). Cached since there are only two multiplier pairs
    in practice (M2's 2x/5x and the STD-based 1x/3x).
    """
    if good_mult <= 0 or warning_mult <= 0 or good_mult >= warning_mult:
        raise ValueError(
            f"Require 0 < good_mult < warning_mult, got good_mult={good_mult}, "
            f"warning_mult={warning_mult}"
        )
    if not (0 < boundary_score < 100):
        raise ValueError(f"boundary_score must be in (0, 100), got {boundary_score}")

    target = (100.0 / boundary_score) - 1.0  # = (good_mult/warning_mult)^p
    ratio = good_mult / warning_mult          # < 1, so ln(ratio) < 0
    p = math.log(target) / math.log(ratio)
    return p


def multiplier_score(value_px: float, floor_px: float, good_mult: float, warning_mult: float) -> float:
    """
    Map a raw px value to a 0-100 score using the generalized-logistic curve
    described in the module docstring, given the GOOD/WARNING multiplier
    pair for whichever metric this value came from.

    Returns NaN if value_px or floor_px is NaN/non-finite/non-positive
    (floor_px), signaling "not scoreable" (e.g. the underlying metric
    FAILed) rather than silently producing a misleading number.
    """
    if floor_px is None or not math.isfinite(floor_px) or floor_px <= 0:
        return float("nan")
    if value_px is None or not math.isfinite(value_px):
        return float("nan")
    if value_px < 0:
        raise ValueError(f"value_px must be non-negative, got {value_px}")

    r = value_px / floor_px
    p = _solve_exponent(good_mult, warning_mult)
    score = 100.0 / (1.0 + (r / warning_mult) ** p)
    return float(min(100.0, max(0.0, score)))


def score_m2(mean_px: float, floor_px: float) -> float:
    """Score for M2 (Edge Alignment) or any per-point-offset metric using
    the M2 GOOD/WARNING multiplier scheme (2x / 5x)."""
    return multiplier_score(mean_px, floor_px, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)


def score_std(std_px: float, floor_px: float) -> float:
    """Score for M3 (Hold-out Consistency) or M4 (Multi-frame Consistency),
    which classify on STD using the 1x / 3x multiplier scheme."""
    return multiplier_score(std_px, floor_px, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)


def score_to_classification(score: float) -> str:
    """
    Inverse mapping: given a 0-100 score (however derived), return the
    GOOD/WARNING/BAD/FAIL label using the same 80/50 boundaries the score
    curve was anchored to. NaN -> "FAIL".
    """
    if score is None or not math.isfinite(score):
        return "FAIL"
    if score >= GOOD_WARNING_BOUNDARY_SCORE:
        return "GOOD"
    if score >= WARNING_BAD_BOUNDARY_SCORE:
        return "WARNING"
    return "BAD"


@dataclass
class NormalizedScore:
    value_px: float
    floor_px: float
    ratio: float          # value_px / floor_px, NaN if not scoreable
    score: float           # 0-100, NaN if not scoreable
    classification: str    # GOOD | WARNING | BAD | FAIL


def normalize(value_px: float, floor_px: float, good_mult: float, warning_mult: float) -> NormalizedScore:
    """Convenience wrapper bundling the ratio, score, and classification
    together for report/UI consumption."""
    score = multiplier_score(value_px, floor_px, good_mult, warning_mult)
    ratio = (value_px / floor_px) if (floor_px and math.isfinite(floor_px) and floor_px > 0
                                       and value_px is not None and math.isfinite(value_px)) else float("nan")
    return NormalizedScore(
        value_px=value_px,
        floor_px=floor_px,
        ratio=ratio,
        score=score,
        classification=score_to_classification(score),
    )
