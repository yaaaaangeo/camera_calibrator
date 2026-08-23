"""
camera_calibrator.calibration.diagnosis
=======================================

Calibration metric들을 사람이 읽는 failure pattern과 recommendation으로 매핑한다.
"""

from __future__ import annotations

import math

from calibration.models.common import regional_edge_average
from calibration.types import (
    CalibrationResult,
    CaptureRecommendation,
    CoverageCell,
    DiagnosisReport,
    DiagnosisSeverity,
    DiversityScores,
    FailurePattern,
    RadialErrorProfile,
    ValidationResult,
)


def _fmt(v: float | None, unit: str = "px") -> str:
    return f"{v:.3f}{unit}" if v is not None else "N/A"


def _add(
    patterns: list[FailurePattern],
    code: str,
    severity: DiagnosisSeverity,
    title: str,
    evidence: list[str],
    recommendation: str,
) -> None:
    patterns.append(FailurePattern(code, severity, title, evidence, recommendation))


def _diagnose_coverage(patterns: list[FailurePattern], dataset_coverage_pct: float | None) -> None:
    if dataset_coverage_pct is None:
        return
    if dataset_coverage_pct < 50.0:
        _add(
            patterns, "poor_coverage", DiagnosisSeverity.ERROR, "Insufficient image coverage",
            [f"Coverage is {dataset_coverage_pct:.1f}%."],
            "Capture more frames with the target near all image edges and corners.",
        )
    elif dataset_coverage_pct < 75.0:
        _add(
            patterns, "limited_coverage", DiagnosisSeverity.WARNING, "Limited image coverage",
            [f"Coverage is {dataset_coverage_pct:.1f}%."],
            "Add a few more views in under-covered image regions before trusting edge behavior.",
        )


def _grid_shape(cells: list[CoverageCell]) -> tuple[int, int]:
    if not cells:
        return 0, 0
    return max(c.row for c in cells) + 1, max(c.col for c in cells) + 1


def _coverage_region_label(row: int, col: int, rows: int, cols: int) -> str:
    v = row / max(rows - 1, 1)
    h = col / max(cols - 1, 1)
    vlabel = "Top" if v < 1 / 3 else ("Bottom" if v > 2 / 3 else "Center")
    hlabel = "Left" if h < 1 / 3 else ("Right" if h > 2 / 3 else "Center")
    if vlabel == "Center" and hlabel == "Center":
        return "Center"
    if vlabel == "Center":
        return hlabel
    if hlabel == "Center":
        return vlabel
    return f"{vlabel}-{hlabel.lower()}"


def _coverage_location_gaps(
    cells: list[CoverageCell],
    low_threshold: float = 0.3,
) -> list[tuple[str, float, int]]:
    rows, cols = _grid_shape(cells)
    if rows == 0 or cols == 0:
        return []
    groups: dict[str, list[CoverageCell]] = {}
    for cell in cells:
        label = _coverage_region_label(cell.row, cell.col, rows, cols)
        groups.setdefault(label, []).append(cell)

    gaps: list[tuple[str, float, int]] = []
    for label, group in groups.items():
        avg_score = sum(c.coverage_score for c in group) / len(group)
        total_corners = sum(c.corner_count for c in group)
        if avg_score < low_threshold:
            gaps.append((label, avg_score, total_corners))

    def sort_key(item: tuple[str, float, int]) -> tuple[float, int, str]:
        label, score, corners = item
        return score, corners, label

    return sorted(gaps, key=sort_key)


def _diagnose_coverage_locations(
    patterns: list[FailurePattern],
    coverage_grid: list[CoverageCell] | None,
) -> None:
    if not coverage_grid:
        return
    gaps = _coverage_location_gaps(coverage_grid)
    if not gaps:
        return
    evidence = [
        f"{label} LOW (coverage {score * 100:.0f}%, corners {corners})"
        for label, score, corners in gaps
    ]
    locations = ", ".join(label for label, _, _ in gaps[:5])
    _add(
        patterns,
        "coverage_location_gaps",
        DiagnosisSeverity.WARNING,
        "Specific image regions are under-covered",
        evidence,
        f"Capture additional target views in these regions: {locations}.",
    )


def _capture_region_name(label: str) -> str:
    """Convert diagnostic grid names to action-oriented capture names."""
    return {
        "Top-left": "Upper-left",
        "Top": "Upper",
        "Top-right": "Upper-right",
    }.get(label, label)


def _add_capture_recommendation(
    recommendations: list[CaptureRecommendation],
    seen: set[str],
    code: str,
    priority: str,
    title: str,
    action: str,
    reason: str,
) -> None:
    if code in seen:
        return
    seen.add(code)
    recommendations.append(CaptureRecommendation(code, priority, title, action, reason))


def _next_capture_recommendations(
    coverage_grid: list[CoverageCell] | None,
    diversity: DiversityScores | None,
) -> list[CaptureRecommendation]:
    recommendations: list[CaptureRecommendation] = []
    seen: set[str] = set()

    if coverage_grid:
        for label, score, corners in _coverage_location_gaps(coverage_grid)[:5]:
            capture_label = _capture_region_name(label)
            priority = "high" if score < 0.10 else "medium"
            _add_capture_recommendation(
                recommendations,
                seen,
                f"capture_{label.lower().replace('-', '_')}",
                priority,
                f"Add {capture_label} board views",
                f"Place the board in the {capture_label} image region and capture 2-3 sharp frames.",
                f"{label} coverage is {score * 100:.0f}% with {corners} detected corners.",
            )

    if diversity is None:
        return recommendations

    if diversity.rotation_diversity < 0.45:
        _add_capture_recommendation(
            recommendations,
            seen,
            "capture_tilt_20_30",
            "high" if diversity.rotation_diversity < 0.30 else "medium",
            "Add 20-30 degree tilted board views",
            "Capture 3-5 frames with the board tilted about 20-30 degrees around yaw and pitch.",
            f"Rotation diversity is {diversity.rotation_diversity * 100:.0f}%.",
        )
    if diversity.distance_diversity < 0.45:
        _add_capture_recommendation(
            recommendations,
            seen,
            "capture_close_distance_board",
            "medium",
            "Add close-distance board views",
            "Capture 2-4 close-distance frames where the board fills more of the image without clipping.",
            f"Distance diversity is {diversity.distance_diversity * 100:.0f}%.",
        )
    if diversity.edge_coverage < 0.55:
        _add_capture_recommendation(
            recommendations,
            seen,
            "capture_edges_and_corners",
            "medium",
            "Add edge and corner board views",
            "Move the board near the image edges and corners, keeping the full pattern visible.",
            f"Edge coverage is {diversity.edge_coverage * 100:.0f}%.",
        )
    if diversity.position_coverage < 0.55 and not coverage_grid:
        _add_capture_recommendation(
            recommendations,
            seen,
            "capture_off_center_positions",
            "medium",
            "Add off-center board positions",
            "Capture frames with the board centered in each side and corner region of the image.",
            f"Position coverage is {diversity.position_coverage * 100:.0f}%.",
        )

    return recommendations


def _edge_rms(cal: CalibrationResult, val: ValidationResult | None) -> float | None:
    if val and val.success and val.edge_rms is not None:
        return val.edge_rms
    if cal.regional_error:
        return regional_edge_average(cal.regional_error)
    return None


def _diagnose_edge_residuals(
    patterns: list[FailurePattern],
    cal: CalibrationResult,
    val: ValidationResult | None,
) -> None:
    edge = _edge_rms(cal, val)
    train = cal.rms_error
    if edge is None:
        return
    if edge > 1.0 or (train is not None and edge > max(train * 1.75, train + 0.35)):
        evidence = [f"Edge RMS is {_fmt(edge)}."]
        if train is not None:
            evidence.append(f"Train RMS is {_fmt(train)}.")
        _add(
            patterns, "edge_residual_high", DiagnosisSeverity.WARNING, "Residuals are high near image edges",
            evidence,
            "Prefer wider pose/edge coverage, then compare Extended Pinhole and Fisheye before exporting.",
        )


def _band_metric(profile: RadialErrorProfile, labels: set[str]) -> float | None:
    values = []
    for b in profile.bins:
        label = (b.label or "").lower()
        if label in labels:
            value = b.p95_error if b.p95_error is not None else b.rms_error
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _diagnose_radial_residuals(patterns: list[FailurePattern], cal: CalibrationResult) -> None:
    profile = cal.radial_bands
    if not profile or not profile.bins:
        return
    center = _band_metric(profile, {"center", "inner"})
    outer = _band_metric(profile, {"outer", "edge", "corner"})
    if center is None or outer is None:
        return
    if outer > max(center * 1.8, center + 0.35):
        _add(
            patterns, "radial_edge_pattern", DiagnosisSeverity.WARNING, "Radial residual grows toward the edge",
            [f"Center/inner residual is {_fmt(center)}.", f"Outer/edge/corner residual is {_fmt(outer)}."],
            "Use a distortion model with enough radial freedom, or capture more tilted/off-axis target poses.",
        )


def _diagnose_train_test_gap(
    patterns: list[FailurePattern],
    cal: CalibrationResult,
    val: ValidationResult | None,
) -> None:
    if not val or not val.success or val.test_rms is None or cal.rms_error is None:
        return
    gap = val.test_rms - cal.rms_error
    ratio = val.test_rms / max(cal.rms_error, 1e-12)
    if gap > 0.30 and ratio > 1.50:
        _add(
            patterns, "train_test_gap", DiagnosisSeverity.WARNING, "Hold-out error is much worse than train error",
            [f"Train RMS is {_fmt(cal.rms_error)}.", f"Test RMS is {_fmt(val.test_rms)}.", f"Gap is {_fmt(gap)}."],
            "Check for overfitting, inconsistent detections, or a train/test split that misses important poses.",
        )


def _diagnose_bootstrap_stability(patterns: list[FailurePattern], cal: CalibrationResult) -> None:
    pu = cal.param_uncertainty_bootstrap or cal.param_uncertainty
    if not pu:
        return
    evidence: list[str] = []
    if pu.overall_stability is not None and pu.overall_stability < 80.0:
        evidence.append(f"Overall parameter stability is {pu.overall_stability:.1f}%.")
    if pu.fx_std is not None and cal.camera_matrix is not None:
        fx = float(cal.camera_matrix[0, 0])
        if fx > 0 and pu.fx_std / fx > 0.01:
            evidence.append(f"fx std is {pu.fx_std:.3f}px ({pu.fx_std / fx * 100:.2f}%).")
    if pu.fy_std is not None and cal.camera_matrix is not None:
        fy = float(cal.camera_matrix[1, 1])
        if fy > 0 and pu.fy_std / fy > 0.01:
            evidence.append(f"fy std is {pu.fy_std:.3f}px ({pu.fy_std / fy * 100:.2f}%).")
    if evidence:
        _add(
            patterns, "unstable_parameters", DiagnosisSeverity.WARNING, "Intrinsic parameters are unstable",
            evidence,
            "Collect more diverse views and re-run bootstrap; avoid relying on tiny distortion differences.",
        )


def _diagnose_observability(patterns: list[FailurePattern], cal: CalibrationResult) -> None:
    obs = cal.observability
    if not obs:
        return
    if obs.rank < obs.jacobian_cols:
        _add(
            patterns, "rank_deficient_observability", DiagnosisSeverity.ERROR, "Some parameters are not observable",
            [f"Jacobian rank is {obs.rank}/{obs.jacobian_cols}."],
            "Reduce model complexity or capture poses that excite the missing parameters.",
        )
    if obs.condition_number is not None and (math.isinf(obs.condition_number) or obs.condition_number >= 1e8):
        _add(
            patterns, "ill_conditioned_observability", DiagnosisSeverity.WARNING, "Calibration is ill-conditioned",
            [f"Condition number is {obs.condition_number:.3g}."],
            "Add views with stronger depth, tilt, and edge variation; consider a simpler camera model.",
        )
    if obs.max_abs_correlation is not None and obs.max_abs_correlation >= 0.98:
        pair = obs.top_correlations[0] if obs.top_correlations else None
        evidence = [f"Max parameter correlation is {obs.max_abs_correlation:.3f}."]
        if pair:
            evidence.append(f"Strongest pair is {pair.param_a}/{pair.param_b} ({pair.correlation:.3f}).")
        _add(
            patterns, "high_parameter_correlation", DiagnosisSeverity.WARNING, "Parameters are strongly coupled",
            evidence,
            "Treat correlated parameters cautiously; more varied target poses usually separate them better.",
        )
    for warning in obs.warnings:
        if not any(warning in e for p in patterns for e in p.evidence):
            _add(
                patterns, "observability_warning", DiagnosisSeverity.INFO, "Observability warning",
                [warning],
                "Review the Jacobian/SVD section before final model selection.",
            )


def diagnose_calibration(
    calibration: CalibrationResult,
    validation: ValidationResult | None = None,
    dataset_coverage_pct: float | None = None,
    coverage_grid: list[CoverageCell] | None = None,
    diversity: DiversityScores | None = None,
) -> DiagnosisReport:
    patterns: list[FailurePattern] = []
    if not calibration.success:
        _add(
            patterns, "calibration_failed", DiagnosisSeverity.ERROR, "Calibration failed",
            [calibration.error_message or "No calibration result is available."],
            "Fix detection/data issues first, then re-run calibration.",
        )
        return DiagnosisReport(model_name=calibration.model_name, patterns=patterns)

    _diagnose_coverage(patterns, dataset_coverage_pct)
    _diagnose_coverage_locations(patterns, coverage_grid)
    _diagnose_edge_residuals(patterns, calibration, validation)
    _diagnose_radial_residuals(patterns, calibration)
    _diagnose_train_test_gap(patterns, calibration, validation)
    _diagnose_bootstrap_stability(patterns, calibration)
    _diagnose_observability(patterns, calibration)

    if not patterns:
        _add(
            patterns, "no_major_failure_pattern", DiagnosisSeverity.INFO, "No major failure pattern detected",
            ["Coverage, residuals, stability, and observability did not cross warning thresholds."],
            "Proceed with export, while keeping application-specific tolerance requirements in mind.",
        )
    capture_recommendations = _next_capture_recommendations(coverage_grid, diversity)
    return DiagnosisReport(
        model_name=calibration.model_name,
        patterns=patterns,
        capture_recommendations=capture_recommendations,
    )


def format_diagnosis_report(report: DiagnosisReport | None) -> str:
    if report is None:
        return "Diagnosis is not available."
    lines = [f"Diagnosis for {report.model_name.value}:"]
    for pattern in report.patterns:
        lines.append(f"- [{pattern.severity.value.upper()}] {pattern.title} ({pattern.code})")
        if pattern.evidence:
            lines.append("  Evidence: " + " ".join(pattern.evidence))
        if pattern.recommendation:
            lines.append("  Recommendation: " + pattern.recommendation)
    if report.capture_recommendations:
        lines.append("Next capture recommendations:")
        for rec in report.capture_recommendations:
            lines.append(f"- [{rec.priority.upper()}] {rec.title}: {rec.action}")
            if rec.reason:
                lines.append(f"  Reason: {rec.reason}")
    return "\n".join(lines)
