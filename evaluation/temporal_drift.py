"""
evaluation/temporal_drift.py

Advanced/Phase-5 metric: Temporal Drift (see evaluation_metric_spec.md
section 17's Stability category). Not part of the MVP scored set.

Complements M4 (Multi-frame Consistency): M4 measures SPREAD (STD/P95/Max)
and flags individual OUTLIER frames, but says nothing about DIRECTION --
a calibration whose error climbs steadily across the sequence (e.g. from
rig flex or thermal drift) can have a perfectly unremarkable STD while
still being on a clear downward trajectory. This metric fits a linear
trend to M4's per-frame error sequence and reports whether that trend is
statistically distinguishable from noise.

Computed directly from an already-produced MultiFrameConsistencyResult
(no new projections needed), so this is cheap to compute given M4 has
already run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from quality.noise_floor import classify, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER


DEFAULT_ALPHA = 0.05
DEFAULT_MIN_FRAMES = 5


@dataclass
class TemporalDriftResult:
    classification: str    # GOOD | WARNING | BAD | FAIL
    slope_px_per_frame: float
    r_value: float
    p_value: float
    is_statistically_significant: bool
    total_drift_px: float   # |slope| * frame span, i.e. predicted total change over the sequence
    floor_px: float
    num_frames_used: int
    warnings: list[str] = field(default_factory=list)


def evaluate_temporal_drift(
    m4_result,
    alpha: float = DEFAULT_ALPHA,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> TemporalDriftResult:
    """
    Fit a linear trend (scipy.stats.linregress) to m4_result's per-frame
    mean_px sequence (frame_index on x). Classification logic:

      - If the trend is NOT statistically significant (p >= alpha): GOOD.
        A slope with no statistical support could just be noise; escalating
        on an insignificant trend would be crying wolf.
      - If significant: classify total predicted drift (|slope| * frame
        span) against floor(Z) using the same STD multiplier scheme M3/M4
        use (1x/3x), since this is also a "how much does behavior vary"
        question, just measured as a directional trend rather than spread.

    FAILs if fewer than min_frames valid (non-FAIL) frames are available
    in m4_result -- not enough points for a meaningful regression.
    """
    valid = [f for f in m4_result.frame_results if f.classification != "FAIL"]
    warnings: list[str] = []

    if len(valid) < min_frames:
        warnings.append(
            f"Only {len(valid)} valid frame(s) (need >= {min_frames}) to fit a temporal trend."
        )
        return TemporalDriftResult(
            classification="FAIL", slope_px_per_frame=float("nan"), r_value=float("nan"),
            p_value=float("nan"), is_statistically_significant=False, total_drift_px=float("nan"),
            floor_px=float("nan"), num_frames_used=len(valid), warnings=warnings,
        )

    x = np.array([f.frame_index for f in valid], dtype=float)
    y = np.array([f.mean_px for f in valid], dtype=float)

    if np.allclose(x, x[0]):
        warnings.append("All valid frames share the same frame_index; cannot fit a trend.")
        return TemporalDriftResult(
            classification="FAIL", slope_px_per_frame=float("nan"), r_value=float("nan"),
            p_value=float("nan"), is_statistically_significant=False, total_drift_px=float("nan"),
            floor_px=float("nan"), num_frames_used=len(valid), warnings=warnings,
        )

    regression = stats.linregress(x, y)
    slope, r_value, p_value = regression.slope, regression.rvalue, regression.pvalue

    frame_span = float(x.max() - x.min())
    total_drift_px = abs(slope) * frame_span
    floor_px = m4_result.floor_px

    significant = bool(p_value < alpha)

    if not significant:
        classification = "GOOD"
    else:
        classification = classify(total_drift_px, floor_px, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)
        direction = "increasing" if slope > 0 else "decreasing"
        warnings.append(
            f"Statistically significant {direction} error trend detected (p={p_value:.4f}): "
            f"~{total_drift_px:.3f}px predicted change across the sequence."
        )

    return TemporalDriftResult(
        classification=classification,
        slope_px_per_frame=float(slope),
        r_value=float(r_value),
        p_value=float(p_value),
        is_statistically_significant=significant,
        total_drift_px=total_drift_px,
        floor_px=floor_px,
        num_frames_used=len(valid),
        warnings=warnings,
    )
