"""
camera_calibrator.export.report
===================================

설계 문서 11번, 12번 - HTML 종합 리포트.

    "OpenCV YAML / ROS camera_info / JSON / CSV / HTML·PDF 리포트까지
     지원하면 실무성이 크게 올라간다."
    "RMS 숫자 하나로 끝내지 말고 종합 리포트를 제공한다."

PDF는 별도 렌더링 엔진(weasyprint, reportlab 등)이 필요하고 시스템 의존성이
붙어 크로스플랫폼 배포(Windows/macOS/Linux, README 1번)에 리스크가 크다.
대신 자체완결형 HTML(외부 리소스 링크 없음)로 만들어서, 필요하면 사용자가
브라우저의 "인쇄 -> PDF로 저장"으로 바로 PDF를 얻을 수 있게 한다 - 실무에서
흔히 쓰이는 절충안이다.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    CrossDatasetValidationResult,
    Dataset,
    FinalResult,
    KFoldResult,
    ModelScore,
    PatternConfig,
    QualityGrade,
    RepeatedKFoldResult,
    ValidationResult,
)
from calibration.models.common import regional_edge_average, distortion_coeff_labels
from calibration.quality import coverage_percentage
from calibration.sanity_check import run_sanity_check, SanitySeverity
from calibration.residual_stats import compute_cdf
from calibration.spatial_error_map import has_systematic_direction_bias, _direction_arrow

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]

_GRADE_STARS = {
    QualityGrade.EXCELLENT: 5,
    QualityGrade.VERY_GOOD: 4,
    QualityGrade.GOOD: 3,
    QualityGrade.WARNING: 2,
    QualityGrade.POOR: 1,
    QualityGrade.REJECT: 0,
}
_GRADE_LABEL_EN = {
    QualityGrade.EXCELLENT: "Excellent",
    QualityGrade.VERY_GOOD: "Very Good",
    QualityGrade.GOOD: "Good",
    QualityGrade.WARNING: "Warning",
    QualityGrade.POOR: "Poor",
    QualityGrade.REJECT: "Reject",
}
_GRADE_COLOR = {
    QualityGrade.EXCELLENT: "#2e7d32",
    QualityGrade.VERY_GOOD: "#558b2f",
    QualityGrade.GOOD: "#9e9d24",
    QualityGrade.WARNING: "#ef6c00",
    QualityGrade.POOR: "#d84315",
    QualityGrade.REJECT: "#c62828",
}


def _fmt(v: float | None, unit: str = "px", digits: int = 3) -> str:
    return f"{v:.{digits}f}{unit}" if v is not None else "N/A"


def _esc(s: object) -> str:
    return html.escape(str(s))


def _fov_deg(focal_px: float, size_px: int) -> float | None:
    """fx(또는 fy)와 이미지 가로(또는 세로) 해상도로 화각(도) 역산.
    2*atan(size / (2*f)) - 설계 문서 12번 리포트 예시의 HFOV/VFOV 항목.
    """
    if focal_px <= 0 or size_px <= 0:
        return None
    return float(math.degrees(2 * math.atan(size_px / (2 * focal_px))))


# ---------------------------------------------------------------------------
# 섹션별 HTML 조각 생성
# ---------------------------------------------------------------------------

def _section_camera_pattern(
    camera_config: CameraConfig, pattern_config: PatternConfig, cal: CalibrationResult | None
) -> str:
    hfov = vfov = None
    if cal and cal.success and cal.camera_matrix is not None:
        fx, fy = float(cal.camera_matrix[0, 0]), float(cal.camera_matrix[1, 1])
        hfov = _fov_deg(fx, camera_config.width)
        vfov = _fov_deg(fy, camera_config.height)

    rows = [
        ("Resolution", f"{camera_config.width} x {camera_config.height}"),
        ("HFOV / VFOV", f"{hfov:.1f} deg / {vfov:.1f} deg" if hfov and vfov else "N/A (캘리브레이션 결과 필요)"),
        ("Pattern", f"{pattern_config.type.value} ({pattern_config.squares_x}x{pattern_config.squares_y} squares)"),
        (
            "Square / Marker size",
            f"{pattern_config.square_size*1000:.1f}mm / {(pattern_config.marker_size or 0)*1000:.1f}mm",
        ),
        ("Dictionary", pattern_config.dictionary or "-"),
    ]
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def _section_dataset(dataset: Dataset) -> str:
    total, detected, enabled = dataset.num_total, dataset.num_detected, dataset.num_enabled
    coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None

    bars_html = ""
    if dataset.diversity:
        d = dataset.diversity
        items = [
            ("Position Coverage", d.position_coverage),
            ("Distance Diversity", d.distance_diversity),
            ("Rotation Diversity", d.rotation_diversity),
            ("Edge Coverage", d.edge_coverage),
            ("Overall Dataset Quality", d.overall),
        ]
        rows = "".join(
            f'<div class="bar-row"><span class="bar-label">{_esc(name)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{val*100:.0f}%"></div></div>'
            f'<span class="bar-pct">{val*100:.0f}%</span></div>'
            for name, val in items
        )
        bars_html = f'<div class="bars">{rows}</div>'

    summary = (
        f"총 {total}장 | 검출 성공 {detected}장 | 사용 중 {enabled}장"
        + (f" | Coverage {coverage_pct:.0f}%" if coverage_pct is not None else "")
    )

    quality_html = ""
    if dataset.quality_score:
        q = dataset.quality_score
        quality_html = (
            '<h4>Overall Dataset Score (설계 문서 4번)</h4>'
            '<table class="kv"><tbody>'
            f'<tr><th>Avg Frame Quality</th><td>{q.avg_frame_quality:.1f}</td></tr>'
            f'<tr><th>Detection Success Rate</th><td>{q.detection_success_rate:.1f}%</td></tr>'
            f'<tr><th>Coverage</th><td>{q.coverage_score:.1f}</td></tr>'
            f'<tr><th>Pose Diversity</th><td>{q.diversity_score:.1f}</td></tr>'
            f'<tr><th>Duplicate Penalty</th><td>-{q.duplicate_penalty:.1f}</td></tr>'
            f'<tr><th><b>Total</b></th><td><b>{q.overall:.1f} ({_esc(q.grade.value)})</b></td></tr>'
            "</tbody></table>"
        )

    return f"<p>{_esc(summary)}</p>{bars_html}{quality_html}"


def _section_detection_statistics(dataset: Dataset) -> str:
    total = dataset.num_total
    detected = dataset.num_detected
    enabled = dataset.num_enabled
    failed = total - detected
    success_rate = detected / total * 100.0 if total else 0.0
    corners = [
        f.detection.num_corners for f in dataset.frames
        if f.detection and f.detection.success
    ]
    areas = [
        f.detection.board_area_ratio for f in dataset.frames
        if f.detection and f.detection.success and f.detection.board_area_ratio is not None
    ]
    tilts = [
        f.detection.board_tilt_deg for f in dataset.frames
        if f.detection and f.detection.success and f.detection.board_tilt_deg is not None
    ]
    failures: dict[str, int] = {}
    for f in dataset.frames:
        if f.detection and not f.detection.success:
            reason = f.detection.failure_reason or "unknown"
            failures[reason] = failures.get(reason, 0) + 1

    def avg(values: list[float | int]) -> str:
        return f"{float(np.mean(values)):.2f}" if values else "N/A"

    rows = [
        ("Total frames", str(total)),
        ("Detected frames", str(detected)),
        ("Enabled frames", str(enabled)),
        ("Failed detections", str(failed)),
        ("Detection success rate", f"{success_rate:.1f}%"),
        ("Avg detected corners", avg(corners)),
        ("Avg board area ratio", f"{float(np.mean(areas)) * 100:.1f}%" if areas else "N/A"),
        ("Avg board tilt", f"{float(np.mean(tilts)):.1f} deg" if tilts else "N/A"),
    ]
    html_rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    html = f'<table class="kv"><tbody>{html_rows}</tbody></table>'

    if dataset.quality_score:
        q = dataset.quality_score
        q_rows = [
            ("Avg Frame Quality", q.avg_frame_quality),
            ("Detection Success", q.detection_success_rate),
            ("Coverage", q.coverage_score),
            ("Pose Diversity", q.diversity_score),
            ("Duplicate Penalty", -q.duplicate_penalty),
            ("Overall Dataset Quality", q.overall),
        ]
        html += "<h4>Full Dataset Quality</h4><table class=\"kv\"><tbody>"
        html += "".join(f"<tr><th>{_esc(k)}</th><td>{v:.1f}</td></tr>" for k, v in q_rows)
        html += f"<tr><th>Grade</th><td>{_esc(q.grade.value.upper())}</td></tr></tbody></table>"

    if failures:
        items = "".join(f"<li>{_esc(reason)}: {count}</li>" for reason, count in sorted(failures.items()))
        html += f"<h4>Detection Failure Reasons</h4><ul>{items}</ul>"
    return html


def _section_model_comparison(
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    scores: list[ModelScore],
    chosen_model: CameraModelType,
) -> str:
    score_by_model = {s.model_name: s for s in scores}
    header = "".join(f"<th>{_esc(_MODEL_LABELS[m])}</th>" for m in _MODEL_ORDER)

    def row(label: str, values: list[str]) -> str:
        cells = "".join(f"<td>{_esc(v)}</td>" for v in values)
        return f"<tr><th>{_esc(label)}</th>{cells}</tr>"

    train_vals, test_vals, edge_vals, straight_vals = [], [], [], []
    aic_vals, bic_vals, score_vals, confidence_vals, chosen_vals = [], [], [], [], []
    for m in _MODEL_ORDER:
        cal = calibration_results.get(m)
        val = validation_results.get(m)
        score = score_by_model.get(m)

        train_vals.append(_fmt(cal.rms_error) if cal and cal.success else "FAIL")
        test_vals.append(_fmt(val.test_rms) if val and val.success else "N/A")
        if val and val.success and val.edge_rms is not None:
            edge_vals.append(_fmt(val.edge_rms))
        elif cal and cal.success and cal.regional_error:
            edge_vals.append(_fmt(regional_edge_average(cal.regional_error)))
        else:
            edge_vals.append("N/A")
        straight_vals.append(_fmt(val.straightness_residual) if val else "N/A")
        aic_vals.append(f"{score.aic:.1f}" if score and score.aic is not None else "N/A")
        bic_vals.append(f"{score.bic:.1f}" if score and score.bic is not None else "N/A")
        score_vals.append(f"{score.score:.3f}" if score else "N/A")
        confidence_vals.append(
            f"{score.selection_confidence_level} {score.selection_confidence:.0f}%"
            if score and score.selection_confidence_level and score.selection_confidence is not None
            else ""
        )

        marks = []
        if score and score.is_recommended:
            marks.append("[recommended]")
        if m == chosen_model:
            marks.append("[chosen]")
        chosen_vals.append(" ".join(marks) if marks else "")

    table = (
        '<table class="compare"><thead><tr><th></th>' + header + "</tr></thead><tbody>"
        + row("Train RMS", train_vals)
        + row("Test RMS", test_vals)
        + row("Edge RMS", edge_vals)
        + row("Straightness", straight_vals)
        + row("AIC", aic_vals)
        + row("BIC", bic_vals)
        + row("Model Score", score_vals)
        + row("Confidence", confidence_vals)
        + row("", chosen_vals)
        + "</tbody></table>"
    )
    recommended = next((s for s in scores if s.is_recommended), None)
    if recommended and recommended.selection_reasons:
        items = "".join(f"<li>{_esc(reason)}</li>" for reason in recommended.selection_reasons)
        table += f"<h4>Recommendation Reasons</h4><ul>{items}</ul>"
    if (
        recommended
        and recommended.selection_confidence_level == "LOW"
        and recommended.selection_confidence_reason
    ):
        table += (
            '<p style="color:#ef6c00;">&#9888; Model selection confidence: LOW - '
            f"{_esc(recommended.selection_confidence_reason)}</p>"
        )
    return table


def _section_cross_validation(
    validation_results: dict[CameraModelType, ValidationResult],
    kfold_result: KFoldResult | None = None,
    repeated_kfold_result: RepeatedKFoldResult | None = None,
) -> str:
    if not validation_results and kfold_result is None and repeated_kfold_result is None:
        return "<p>Cross validation is not available.</p>"

    rows = ""
    for model in _MODEL_ORDER:
        val = validation_results.get(model)
        if val is None:
            continue
        train_p95 = val.train_residual_stats.p95 if val.train_residual_stats else None
        test_p95 = val.test_residual_stats.p95 if val.test_residual_stats else None
        gap = val.test_rms - val.train_rms if val.test_rms is not None and val.train_rms is not None else None
        rows += (
            "<tr>"
            f"<td>{_esc(_MODEL_LABELS[model])}</td>"
            f"<td>{_esc('OK' if val.success else 'FAIL')}</td>"
            f"<td>{len(val.train_frame_ids)}</td><td>{len(val.test_frame_ids)}</td>"
            f"<td>{_fmt(val.train_rms)}</td><td>{_fmt(val.test_rms)}</td>"
            f"<td>{_fmt(train_p95)}</td><td>{_fmt(test_p95)}</td>"
            f"<td>{_fmt(gap)}</td><td>{_fmt(val.edge_rms)}</td>"
            f"<td>{_fmt(val.straightness_residual)}</td>"
            "</tr>"
        )
    html = (
        "<h4>Hold-out Cross Validation</h4>"
        '<table class="compare"><thead><tr><th>Model</th><th>Status</th><th>Train Frames</th>'
        '<th>Test Frames</th><th>Train RMS</th><th>Test RMS</th><th>Train P95</th>'
        '<th>Test P95</th><th>Gap</th><th>Edge RMS</th><th>Straightness</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
        if rows else "<p>Hold-out validation is not available.</p>"
    )

    if kfold_result:
        html += (
            "<h4>K-Fold Cross Validation</h4>"
            '<table class="kv"><tbody>'
            f"<tr><th>K</th><td>{kfold_result.k}</td></tr>"
            f"<tr><th>Successful folds</th><td>{kfold_result.n_successful_folds}/{kfold_result.k}</td></tr>"
            f"<tr><th>Mean Test RMS</th><td>{_fmt(kfold_result.mean_test_rms)}</td></tr>"
            f"<tr><th>Std Test RMS</th><td>{_fmt(kfold_result.std_test_rms)}</td></tr>"
            f"<tr><th>Mean Test P95</th><td>{_fmt(kfold_result.mean_test_p95)}</td></tr>"
            "</tbody></table>"
        )
    else:
        html += '<p style="color:#777; font-size:12px;">K-Fold Cross Validation was not attached to this report.</p>'

    if repeated_kfold_result:
        html += (
            "<h4>Repeated K-Fold Cross Validation</h4>"
            '<table class="kv"><tbody>'
            f"<tr><th>K x repeats</th><td>{repeated_kfold_result.k} x {repeated_kfold_result.n_repeats}</td></tr>"
            f"<tr><th>Successful runs</th><td>{repeated_kfold_result.n_successful_runs}</td></tr>"
            f"<tr><th>Mean Test RMS</th><td>{_fmt(repeated_kfold_result.mean_test_rms)}</td></tr>"
            f"<tr><th>Std Test RMS</th><td>{_fmt(repeated_kfold_result.std_test_rms)}</td></tr>"
            f"<tr><th>Mean Test P95</th><td>{_fmt(repeated_kfold_result.mean_test_p95)}</td></tr>"
            f"<tr><th>Std Test P95</th><td>{_fmt(repeated_kfold_result.std_test_p95)}</td></tr>"
            "</tbody></table>"
        )
    return html


def _section_bootstrap_stability(calibration_results: dict[CameraModelType, CalibrationResult]) -> str:
    rows = ""
    detail_sections = []
    for model in _MODEL_ORDER:
        cal = calibration_results.get(model)
        if cal is None or not cal.success:
            continue
        pu = cal.param_uncertainty_bootstrap or cal.param_uncertainty
        if pu is None:
            rows += f"<tr><td>{_esc(_MODEL_LABELS[model])}</td><td colspan=\"8\">N/A</td></tr>"
            continue
        stability = f"{pu.overall_stability:.1f}/100" if pu.overall_stability is not None else "N/A"
        status = "OK"
        if cal.camera_matrix is not None:
            fx = float(cal.camera_matrix[0, 0])
            fy = float(cal.camera_matrix[1, 1])
            status = "OK" if pu.is_within_threshold(fx, fy) else "WARNING"
        rows += (
            "<tr>"
            f"<td>{_esc(_MODEL_LABELS[model])}</td><td>{_esc(pu.method)}</td>"
            f"<td>{pu.n_bootstrap_success if pu.n_bootstrap_success is not None else 'N/A'}</td>"
            f"<td>{_fmt(pu.fx_std)}</td><td>{_fmt(pu.fy_std)}</td>"
            f"<td>{_fmt(pu.cx_std)}</td><td>{_fmt(pu.cy_std)}</td>"
            f"<td>{_esc(stability)}</td>"
            f"<td>{_esc(status)}</td>"
            "</tr>"
        )
        if pu.method == "bootstrap":
            intrinsic_rows = []
            for name, mean, std, median, min_v, max_v, lo, hi, stability in (
                ("fx", pu.fx_mean, pu.fx_std, pu.fx_median, pu.fx_min, pu.fx_max, pu.fx_ci_low, pu.fx_ci_high, pu.fx_stability),
                ("fy", pu.fy_mean, pu.fy_std, pu.fy_median, pu.fy_min, pu.fy_max, pu.fy_ci_low, pu.fy_ci_high, pu.fy_stability),
                ("cx", pu.cx_mean, pu.cx_std, pu.cx_median, pu.cx_min, pu.cx_max, pu.cx_ci_low, pu.cx_ci_high, pu.cx_stability),
                ("cy", pu.cy_mean, pu.cy_std, pu.cy_median, pu.cy_min, pu.cy_max, pu.cy_ci_low, pu.cy_ci_high, pu.cy_stability),
            ):
                intrinsic_rows.append(
                    f"<tr><td>{name}</td><td>{_fmt(mean)}</td><td>{_fmt(std)}</td>"
                    f"<td>{_fmt(median)}</td><td>{_fmt(min_v)}</td><td>{_fmt(max_v)}</td>"
                    f"<td>{_fmt(lo)} ~ {_fmt(hi)}</td>"
                    f"<td>{stability:.1f}/100</td></tr>" if stability is not None else
                    f"<tr><td>{name}</td><td>{_fmt(mean)}</td><td>{_fmt(std)}</td>"
                    f"<td>{_fmt(median)}</td><td>{_fmt(min_v)}</td><td>{_fmt(max_v)}</td>"
                    f"<td>{_fmt(lo)} ~ {_fmt(hi)}</td><td>N/A</td></tr>"
                )
            details = (
                f"<h4>{_esc(_MODEL_LABELS[model])} Bootstrap Intrinsic Distribution</h4>"
                '<table class="compare"><thead><tr><th>Param</th><th>Mean</th><th>Std</th>'
                '<th>Median</th><th>Min</th><th>Max</th><th>95% CI</th><th>Stability</th>'
                f"</tr></thead><tbody>{''.join(intrinsic_rows)}</tbody></table>"
            )
            if pu.distortion_stats:
                dist_rows = "".join(
                    f"<tr><td>{_esc(stat.label or f'd{stat.index}')}</td>"
                    f"<td>{_fmt(stat.mean)}</td><td>{_fmt(stat.std)}</td>"
                    f"<td>{_fmt(stat.median)}</td><td>{_fmt(stat.min)}</td><td>{_fmt(stat.max)}</td>"
                    f"<td>{_fmt(stat.ci_low)} ~ {_fmt(stat.ci_high)}</td>"
                    f"<td>{stat.stability_score:.1f}/100</td></tr>"
                    if stat.stability_score is not None else
                    f"<tr><td>{_esc(stat.label or f'd{stat.index}')}</td>"
                    f"<td>{_fmt(stat.mean)}</td><td>{_fmt(stat.std)}</td>"
                    f"<td>{_fmt(stat.median)}</td><td>{_fmt(stat.min)}</td><td>{_fmt(stat.max)}</td>"
                    f"<td>{_fmt(stat.ci_low)} ~ {_fmt(stat.ci_high)}</td><td>N/A</td></tr>"
                    for stat in pu.distortion_stats
                )
                details += (
                    f"<h4>{_esc(_MODEL_LABELS[model])} Bootstrap Distortion Coefficients</h4>"
                    '<table class="compare"><thead><tr><th>Coeff</th><th>Mean</th><th>Std</th>'
                    '<th>Median</th><th>Min</th><th>Max</th><th>95% CI</th><th>Stability</th>'
                    f"</tr></thead><tbody>{dist_rows}</tbody></table>"
                )
            detail_sections.append(details)
    if not rows:
        return "<p>Bootstrap stability is not available.</p>"
    return (
        '<table class="compare"><thead><tr><th>Model</th><th>Method</th><th>Bootstrap N</th>'
        '<th>fx std</th><th>fy std</th><th>cx std</th><th>cy std</th>'
        f"<th>Stability</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>"
        + "".join(detail_sections)
    )


def _section_final_summary(final: FinalResult) -> str:
    cal, val, conf = final.calibration, final.validation, final.confidence
    rows = [
        ("Chosen Model", _MODEL_LABELS.get(final.chosen_model, final.chosen_model.value)),
        ("Overall Grade", final.overall_grade.value.upper()),
        ("Final Calibration Confidence", f"{conf.score:.0f}/100 - {conf.level}" if conf else "N/A"),
        ("Train RMS", _fmt(cal.rms_error if cal else None)),
        ("Test RMS", _fmt(val.test_rms if val else None)),
        ("Test P95", _fmt(val.test_residual_stats.p95 if val and val.test_residual_stats else None)),
        ("Edge RMS", _fmt(val.edge_rms if val else None)),
        ("Straightness", _fmt(val.straightness_residual if val else None)),
        ("Dataset Coverage", f"{final.dataset_coverage_pct:.1f}%" if final.dataset_coverage_pct is not None else "N/A"),
        (
            "Observability",
            f"{cal.observability.observability_grade} ({cal.observability.observability_score:.1f}/100)"
            if cal and cal.observability and cal.observability.observability_score is not None else "N/A",
        ),
        (
            "Undistortion Quality",
            f"{cal.undistortion_quality.quality_grade.value.upper()} ({cal.undistortion_quality.quality_score:.1f}/100)"
            if cal and cal.undistortion_quality else "N/A",
        ),
    ]
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def _section_chosen_model_detail(cal: CalibrationResult | None, val: ValidationResult | None) -> str:
    if cal is None or not cal.success:
        return "<p>선택된 모델의 캘리브레이션 결과가 없습니다.</p>"

    K = cal.camera_matrix
    D = cal.distortion.ravel() if cal.distortion is not None else np.array([])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    labels = distortion_coeff_labels(cal.model_name, D.size)
    distortion_str = ", ".join(f"{name}={v:.6f}" for name, v in zip(labels, D.tolist()))

    rows = [
        ("fx, fy", f"{fx:.2f}, {fy:.2f}"),
        ("cx, cy", f"{cx:.2f}, {cy:.2f}"),
        (f"Distortion coeffs ({D.size}개)", distortion_str),
        ("RMS Reprojection Error", _fmt(cal.rms_error)),
    ]
    if cal.param_uncertainty and cal.param_uncertainty.fx_std is not None:
        pu = cal.param_uncertainty
        # 설계 문서 20/21/22번 - method로 covariance/bootstrap 출처를 구분해
        # 보여준다 (Fisheye는 OpenCV가 covariance를 안 줘서 원래부터 bootstrap).
        method_note = " (bootstrap 추정)" if pu.method == "bootstrap" else ""
        rows.append((f"fx/fy std dev{method_note}", f"{pu.fx_std:.3f} / {pu.fy_std:.3f}"))
        within = pu.is_within_threshold(fx, fy)
        rows.append(("Parameter Stability", "Good (within 1%)" if within else "Warning (over 1%)"))
        if pu.fx_ci_low is not None and pu.fx_ci_high is not None:
            rows.append(("fx 95% CI", f"{pu.fx_ci_low:.1f} ~ {pu.fx_ci_high:.1f}"))
        if pu.fy_ci_low is not None and pu.fy_ci_high is not None:
            rows.append(("fy 95% CI", f"{pu.fy_ci_low:.1f} ~ {pu.fy_ci_high:.1f}"))

    if cal.param_uncertainty_bootstrap and cal.param_uncertainty_bootstrap.fx_std is not None:
        # 설계 문서 20번 - covariance 기반(위)과 별개로 계산한 교차검증용
        # bootstrap 추정치. 두 값이 크게 다르면 covariance의 선형근사가
        # 이 데이터셋에서 잘 안 맞는다는 신호다.
        pb = cal.param_uncertainty_bootstrap
        rows.append((
            "fx/fy std dev (bootstrap, 교차검증)",
            f"{pb.fx_std:.3f} / {pb.fy_std:.3f}"
            + (f"  (95% CI: {pb.fx_ci_low:.1f}~{pb.fx_ci_high:.1f} / {pb.fy_ci_low:.1f}~{pb.fy_ci_high:.1f})"
               if pb.fx_ci_low is not None else ""),
        ))

    if cal.observability:
        obs = cal.observability
        if obs.observability_score is not None:
            rows.append((
                "Observability",
                f"{obs.observability_grade or 'N/A'} ({obs.observability_score:.1f}/100)",
            ))
        rows.append(("Jacobian", f"{obs.jacobian_rows} x {obs.jacobian_cols} ({obs.num_points} points)"))
        rows.append(("Jacobian Rank", f"{obs.rank} / {obs.jacobian_cols}"))
        rows.append((
            "Condition Number",
            f"{obs.condition_number:.3g}" if obs.condition_number is not None else "N/A",
        ))
        if obs.min_singular_value is not None and obs.max_singular_value is not None:
            rows.append(("Singular Value Range", f"{obs.min_singular_value:.3g} ~ {obs.max_singular_value:.3g}"))
        if obs.max_abs_correlation is not None:
            rows.append(("Max Parameter Correlation", f"{obs.max_abs_correlation:.3f}"))

    if cal.undistortion_quality:
        uq = cal.undistortion_quality
        rows.append((
            "Undistortion Quality",
            f"{uq.quality_grade.value.upper()} ({uq.quality_score:.1f}/100)",
        ))
        rows.append(("Valid Pixel Ratio", f"{uq.valid_pixel_ratio * 100:.1f}%"))
        rows.append(("Black Border Ratio", f"{uq.black_border_ratio * 100:.1f}%"))
        rows.append(("ROI Loss", f"{uq.roi_loss_ratio * 100:.1f}%"))
        rows.append(("Valid ROI", f"x={uq.valid_roi[0]}, y={uq.valid_roi[1]}, w={uq.valid_roi[2]}, h={uq.valid_roi[3]}"))
        if uq.undistorted_black_pixel_ratio is not None:
            sample = f" ({uq.sample_frame_id})" if uq.sample_frame_id else ""
            rows.append(("Undistorted Image Black Pixels" + sample, f"{uq.undistorted_black_pixel_ratio * 100:.1f}%"))

    if cal.regional_error:
        re = cal.regional_error
        rows.append((
            "Regional RMS",
            f"center={_fmt(re.center)} left={_fmt(re.left)} right={_fmt(re.right)} "
            f"top={_fmt(re.top)} bottom={_fmt(re.bottom)} corner={_fmt(re.corner)}",
        ))

    if val and val.success:
        rows.append(("Hold-out Test RMS", _fmt(val.test_rms)))
        rows.append(("Edge RMS (test)", _fmt(val.edge_rms)))
        rows.append(("Line Straightness", _fmt(val.straightness_residual)))

    kv = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    detail = f'<table class="kv"><tbody>{kv}</tbody></table>'

    if val and val.straightness_breakdown and val.straightness_breakdown.num_lines > 0:
        # 설계 문서 15번 - Line Straightness 방향/위치별 분해.
        sb = val.straightness_breakdown
        sb_rows = [
            ("Horizontal", sb.horizontal_error), ("Vertical", sb.vertical_error),
            ("Diagonal", sb.diagonal_error), ("Center line", sb.center_line_error),
            ("Edge line", sb.edge_line_error), ("Corner line", sb.corner_line_error),
            ("Overall", sb.overall_error),
        ]
        sb_kv = "".join(f"<tr><th>{_esc(k)}</th><td>{_fmt(v)}</td></tr>" for k, v in sb_rows)
        detail += (
            f"<h4>Line Straightness Breakdown (n={sb.num_lines} lines)</h4>"
            f'<table class="kv"><tbody>{sb_kv}</tbody></table>'
        )

    if cal.observability:
        obs = cal.observability
        detail += "<h4>Observability (Jacobian / SVD)</h4>"
        if obs.top_correlations:
            corr_rows = "".join(
                f"<tr><td>{_esc(c.param_a)}</td><td>{_esc(c.param_b)}</td>"
                f"<td>{c.correlation:.3f}</td></tr>"
                for c in obs.top_correlations
            )
            detail += (
                '<table class="compare"><thead><tr><th>Parameter A</th><th>Parameter B</th>'
                f"<th>Correlation</th></tr></thead><tbody>{corr_rows}</tbody></table>"
            )
        else:
            detail += "<p>강한 parameter correlation이 감지되지 않았습니다.</p>"
        if obs.correlation_matrix and obs.parameter_labels:
            header = "<th></th>" + "".join(f"<th>{_esc(label)}</th>" for label in obs.parameter_labels)
            matrix_rows = ""
            for label, values in zip(obs.parameter_labels, obs.correlation_matrix):
                cells = "".join(f"<td>{float(v):.3f}</td>" for v in values)
                matrix_rows += f"<tr><th>{_esc(label)}</th>{cells}</tr>"
            detail += (
                "<h4>Parameter Correlation Matrix</h4>"
                f'<table class="compare"><thead><tr>{header}</tr></thead><tbody>{matrix_rows}</tbody></table>'
            )
        if obs.warnings:
            items = "".join(f"<li>{_esc(w)}</li>" for w in obs.warnings)
            detail += f'<ul style="color:#ef6c00;">{items}</ul>'

    if cal.undistortion_quality and cal.undistortion_quality.warnings:
        uq_items = "".join(f"<li>{_esc(w)}</li>" for w in cal.undistortion_quality.warnings)
        detail += f'<h4>Undistortion Quality Warnings</h4><ul style="color:#ef6c00;">{uq_items}</ul>'

    if cal.radial_profile and cal.radial_profile.bins:
        bin_rows = "".join(
            f"<tr><td>{b.radius_min:.0f} ~ {b.radius_max:.0f}px</td>"
            f"<td>{_fmt(b.mean_error)}</td><td>{b.num_points}</td></tr>"
            for b in cal.radial_profile.bins
        )
        detail += (
            "<h4>Edge Error Map (Radial Error Profile)</h4>"
            '<table class="compare"><thead><tr><th>Radius range</th><th>Mean error</th>'
            f"<th>Points</th></tr></thead><tbody>{bin_rows}</tbody></table>"
        )

    if cal.radial_bands and cal.radial_bands.bins:
        # 설계 문서 14번 - Center/Inner/Middle/Outer/Edge/Corner 명명 대역 표.
        band_rows = "".join(
            f"<tr><td>{_esc(b.label)}</td><td>{b.radius_min:.0f} ~ {b.radius_max:.0f}px</td>"
            f"<td>{_fmt(b.mean_error)}</td><td>{_fmt(b.median_error)}</td>"
            f"<td>{_fmt(b.rms_error)}</td><td>{_fmt(b.p95_error)}</td>"
            f"<td>{_fmt(b.max_error)}</td><td>{b.num_points}</td></tr>"
            for b in cal.radial_bands.bins
        )
        detail += (
            "<h4>Radial Error Bands (Center ~ Corner)</h4>"
            '<table class="compare"><thead><tr><th>Band</th><th>Range</th><th>Mean</th>'
            "<th>Median</th><th>RMS</th><th>P95</th><th>Max</th><th>N</th></tr></thead>"
            f"<tbody>{band_rows}</tbody></table>"
        )

    if cal.residual_stats and cal.residual_stats.n > 0:
        rs = cal.residual_stats
        # 설계 문서 11/12번 - 코너 포인트 단위 재투영 오차 percentile/histogram.
        rs_rows = [
            ("N (points)", str(rs.n)),
            ("RMSE", _fmt(rs.rmse)), ("MAE", _fmt(rs.mae)), ("Median", _fmt(rs.median)),
            ("Std", _fmt(rs.std)), ("Min", _fmt(rs.min)), ("Q1", _fmt(rs.q1)), ("Q3", _fmt(rs.q3)),
            ("P90", _fmt(rs.p90)), ("P95", _fmt(rs.p95)), ("P99", _fmt(rs.p99)), ("Max", _fmt(rs.max)),
            ("Outliers (median+3*MAD 초과)", str(rs.outlier_count)),
        ]
        rs_kv = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rs_rows)
        detail += f'<h4>Residual Distribution</h4><table class="kv"><tbody>{rs_kv}</tbody></table>'

        if rs.histogram_counts:
            max_count = max(rs.histogram_counts)
            hist_rows = "".join(
                f"<tr><td>{rs.histogram_bin_edges[i]:.2f} ~ {rs.histogram_bin_edges[i+1]:.2f}px</td>"
                f'<td><div class="bar-track" style="display:inline-block;width:200px;">'
                f'<div class="bar-fill" style="width:{(count/max_count*100) if max_count else 0:.0f}%"></div>'
                f"</div></td><td>{count}</td></tr>"
                for i, count in enumerate(rs.histogram_counts)
            )
            detail += (
                "<h4>Residual Histogram</h4>"
                '<table class="compare"><thead><tr><th>Range</th><th>Distribution</th>'
                f"<th>Count</th></tr></thead><tbody>{hist_rows}</tbody></table>"
            )

        cdf_edges, cdf_values = compute_cdf(rs)
        if cdf_edges:
            # 설계 문서 12번 - CDF. histogram과 같은 bar-track 스타일이지만
            # "구간별 개수"가 아니라 "이 값 이하가 전체의 몇 %인가"를 누적으로 보여준다.
            cdf_rows = "".join(
                f"<tr><td>&le; {edge:.2f}px</td>"
                f'<td><div class="bar-track" style="display:inline-block;width:200px;">'
                f'<div class="bar-fill" style="width:{frac*100:.0f}%"></div>'
                f"</div></td><td>{frac*100:.1f}%</td></tr>"
                for edge, frac in zip(cdf_edges, cdf_values)
            )
            detail += (
                "<h4>CDF (Cumulative Distribution Function)</h4>"
                '<table class="compare"><thead><tr><th>Residual &le;</th><th>Cumulative</th>'
                f"<th>%</th></tr></thead><tbody>{cdf_rows}</tbody></table>"
                f'<p style="color:#777; font-size:12px;">P90={_fmt(rs.p90)} | '
                f"P95={_fmt(rs.p95)} | P99={_fmt(rs.p99)}</p>"
            )

        if rs.min is not None and rs.max is not None and rs.max > rs.min:
            # 설계 문서 12번 - Box Plot을 간단한 SVG로 그린다 (min-max 축, Q1~Q3 박스, median 선).
            svg_w, svg_h = 480, 70
            margin = 20
            span = rs.max - rs.min

            def _x(v: float) -> float:
                return margin + (v - rs.min) / span * (svg_w - 2 * margin)

            x_min, x_q1, x_med, x_q3, x_max = _x(rs.min), _x(rs.q1), _x(rs.median), _x(rs.q3), _x(rs.max)
            mid_y = svg_h / 2
            box_h = 24
            detail += (
                "<h4>Box Plot</h4>"
                f'<svg viewBox="0 0 {svg_w} {svg_h}" width="100%" height="{svg_h}" '
                'xmlns="http://www.w3.org/2000/svg" style="font-size:10px;">'
                f'<line x1="{x_min:.1f}" y1="{mid_y}" x2="{x_max:.1f}" y2="{mid_y}" stroke="#455a64" stroke-width="1.5"/>'
                f'<line x1="{x_min:.1f}" y1="{mid_y-box_h/4}" x2="{x_min:.1f}" y2="{mid_y+box_h/4}" stroke="#455a64" stroke-width="1.5"/>'
                f'<line x1="{x_max:.1f}" y1="{mid_y-box_h/4}" x2="{x_max:.1f}" y2="{mid_y+box_h/4}" stroke="#455a64" stroke-width="1.5"/>'
                f'<rect x="{x_q1:.1f}" y="{mid_y-box_h/2}" width="{max(1.0, x_q3-x_q1):.1f}" height="{box_h}" '
                'fill="#90a4ae" fill-opacity="0.5" stroke="#455a64"/>'
                f'<line x1="{x_med:.1f}" y1="{mid_y-box_h/2}" x2="{x_med:.1f}" y2="{mid_y+box_h/2}" stroke="#c62828" stroke-width="2"/>'
                f'<text x="{x_min:.1f}" y="{svg_h-4}" text-anchor="middle">{rs.min:.2f}</text>'
                f'<text x="{x_q1:.1f}" y="{svg_h-4}" text-anchor="middle">{rs.q1:.2f}</text>'
                f'<text x="{x_med:.1f}" y="14" text-anchor="middle">{rs.median:.2f}</text>'
                f'<text x="{x_q3:.1f}" y="{svg_h-4}" text-anchor="middle">{rs.q3:.2f}</text>'
                f'<text x="{x_max:.1f}" y="{svg_h-4}" text-anchor="middle">{rs.max:.2f}</text>'
                "</svg>"
            )

    if cal.spatial_error_map and cal.spatial_error_map.cells:
        smap = cal.spatial_error_map
        # 설계 문서 13번 - Spatial Error Map (X/Y 방향 heatmap). 격자 각 칸을
        # 색(RMS 크기) + 화살표(평균 오차 방향)로 표시한다.
        grid = {(c.row, c.col): c for c in smap.cells}
        max_rms = max((c.rms for c in smap.cells if c.rms is not None), default=0.0)
        cell_rows = ""
        for r in range(smap.rows):
            cell_cols = ""
            for c in range(smap.cols):
                cell = grid.get((r, c))
                if cell is None or cell.num_points == 0 or cell.rms is None:
                    cell_cols += '<td style="background:#f0f0f0; color:#999;">N/A</td>'
                    continue
                intensity = cell.rms / max_rms if max_rms > 0 else 0.0
                # 초록(오차 작음) -> 빨강(오차 큼) 그라데이션
                red = int(255 * intensity)
                green = int(200 * (1 - intensity))
                arrow = _direction_arrow(cell.direction_deg) if cell.direction_deg is not None else ""
                cell_cols += (
                    f'<td style="background:rgb({red},{green},60); color:#fff; text-align:center;">'
                    f"{arrow} {cell.rms:.2f}<br/><span style=\"font-size:10px;\">P95 {cell.p95:.2f}</span></td>"
                )
            cell_rows += f"<tr>{cell_cols}</tr>"

        bias_warning = ""
        if has_systematic_direction_bias(smap):
            bias_warning = (
                '<p style="color:#ef6c00;">&#9888; 여러 칸의 오차 방향이 한쪽으로 몰려 있습니다 - '
                "카메라 모델이 왜곡을 충분히 설명하지 못하고 있을 가능성이 있습니다.</p>"
            )
        else:
            bias_warning = '<p style="color:#2e7d32;">방향에 뚜렷한 편향이 감지되지 않았습니다.</p>'

        detail += (
            "<h4>Spatial Error Map (X/Y 방향)</h4>"
            f'<table class="compare"><tbody>{cell_rows}</tbody></table>'
            f"{bias_warning}"
        )

    return detail


def _section_sanity_check(camera_config: CameraConfig, cal: CalibrationResult | None) -> str:
    """설계 문서 8번 - Calibration 결과 sanity check 섹션.

    fx/fy 양수 여부, principal point 위치, aspect ratio, distortion 크기,
    FOV, RMS 등을 훑어 물리적으로 이상한 부분이 있으면 경고 목록으로 보여준다.
    경고가 하나도 없어도 "확인했다"는 걸 알 수 있도록 안내 문구를 남긴다.
    """
    if cal is None or not cal.success:
        return "<p>캘리브레이션 결과가 없어 sanity check를 실행할 수 없습니다.</p>"

    check = run_sanity_check(cal, camera_config)
    if not check.issues:
        return '<p style="color:#2e7d32;">&#10003; 이상 없음 - fx/fy, principal point, aspect ratio, distortion, FOV, RMS 모두 통상 범위 안입니다.</p>'

    items = "".join(
        f'<li><span style="color:{"#c62828" if i.severity == SanitySeverity.ERROR else "#ef6c00"};">'
        f'{"[ERROR]" if i.severity == SanitySeverity.ERROR else "[WARNING]"}</span> {_esc(i.message)}</li>'
        for i in check.issues
    )
    return f"<ul>{items}</ul>"


def _section_outlier(final: FinalResult) -> str:
    if not final.outlier or not final.outlier.removed_frame_ids:
        return "<p>제외된 이상치 프레임이 없습니다.</p>"
    o = final.outlier
    items = "".join(f"<li>{_esc(fid)}</li>" for fid in o.removed_frame_ids)
    return (
        f"<p>{o.iterations} iteration(s), threshold={o.threshold_used:.3f}px, "
        f"RMS {_fmt(o.rms_before)} -&gt; {_fmt(o.rms_after)}</p>"
        f"<ul>{items}</ul>"
    )


def _section_diagnosis(final: FinalResult) -> str:
    report = final.diagnosis
    if report is None or not report.patterns:
        return "<p>Diagnosis is not available.</p>"

    def sev_color(severity) -> str:
        return {
            "error": "#c62828",
            "warning": "#ef6c00",
            "info": "#2e7d32",
        }.get(getattr(severity, "value", str(severity)), "#455a64")

    rows = ""
    for pattern in report.patterns:
        evidence = "<br/>".join(_esc(e) for e in pattern.evidence) if pattern.evidence else "-"
        rows += (
            "<tr>"
            f'<td style="color:{sev_color(pattern.severity)}; font-weight:bold;">{_esc(pattern.severity.value.upper())}</td>'
            f"<td>{_esc(pattern.title)}<br/><span style=\"color:#777; font-size:11px;\">{_esc(pattern.code)}</span></td>"
            f"<td>{evidence}</td>"
            f"<td>{_esc(pattern.recommendation)}</td>"
            "</tr>"
        )
    pattern_table = (
        '<table class="compare"><thead><tr><th>Severity</th><th>Pattern</th>'
        f"<th>Evidence</th><th>Recommendation</th></tr></thead><tbody>{rows}</tbody></table>"
    )
    if not report.capture_recommendations:
        return pattern_table

    rec_rows = ""
    for rec in report.capture_recommendations:
        rec_rows += (
            "<tr>"
            f"<td>{_esc(rec.priority.upper())}</td>"
            f"<td>{_esc(rec.title)}<br/><span style=\"color:#777; font-size:11px;\">{_esc(rec.code)}</span></td>"
            f"<td>{_esc(rec.action)}</td>"
            f"<td>{_esc(rec.reason) if rec.reason else '-'}</td>"
            "</tr>"
        )
    rec_table = (
        "<h4>Next Capture Recommendations</h4>"
        '<table class="compare"><thead><tr><th>Priority</th><th>Capture</th>'
        f"<th>Action</th><th>Reason</th></tr></thead><tbody>{rec_rows}</tbody></table>"
    )
    return pattern_table + rec_table


def _section_cross_dataset(results: list[CrossDatasetValidationResult] | None) -> str:
    if not results:
        return "<p>Cross-dataset validation is not available.</p>"

    rows = ""
    for r in results:
        p95 = f"{r.test_p95:.3f}" if r.test_p95 is not None else "N/A"
        train = f"{r.train_rms:.3f}" if r.train_rms is not None else "N/A"
        test = f"{r.test_rms:.3f}" if r.test_rms is not None else "N/A"
        gap = f"{r.generalization_gap:.3f}" if r.generalization_gap is not None else "N/A"
        status = "OK" if r.success else f"FAIL: {r.error_message or ''}"
        rows += (
            "<tr>"
            f"<td>{_esc(r.source_dataset_id)}</td>"
            f"<td>{_esc(r.target_dataset_id)}</td>"
            f"<td>{_esc(_MODEL_LABELS.get(r.model_name, r.model_name.value))}</td>"
            f"<td>{train}</td><td>{test}</td><td>{p95}</td><td>{gap}</td>"
            f"<td>{r.num_test_frames}</td><td>{_esc(status)}</td>"
            "</tr>"
        )
    return (
        '<table class="compare"><thead><tr><th>Source</th><th>Target</th><th>Model</th>'
        '<th>Train RMS</th><th>Target RMS</th><th>Target P95</th><th>Gap</th>'
        f"<th>Frames</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif; margin: 0; padding: 32px;
       color: #1a1a1a; background: #fff; max-width: 900px; margin-left:auto; margin-right:auto; }
h1 { font-size: 22px; border-bottom: 3px solid #1a1a1a; padding-bottom: 8px; }
h2 { font-size: 17px; margin-top: 36px; border-left: 5px solid #455a64; padding-left: 10px; }
h4 { font-size: 14px; margin-top: 18px; color: #444; }
.meta { color: #666; font-size: 13px; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
table.kv th { text-align: left; width: 220px; background: #f5f5f5; }
table th, table td { border: 1px solid #ddd; padding: 6px 10px; }
table.compare th, table.compare td { text-align: center; }
table.compare thead th { background: #37474f; color: #fff; }
table.compare tbody th { text-align: left; background: #f5f5f5; }
.bars { margin: 12px 0; }
.bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 13px; }
.bar-label { width: 190px; }
.bar-track { flex: 1; background: #eee; border-radius: 3px; height: 14px; overflow: hidden; }
.bar-fill { background: #43a047; height: 100%; }
.bar-pct { width: 44px; text-align: right; }
.grade-box { display: inline-block; padding: 14px 22px; border-radius: 6px; color: #fff;
             font-size: 18px; font-weight: bold; margin-top: 10px; }
.stars { font-size: 20px; letter-spacing: 2px; }
@media print { body { padding: 0; } h2 { break-after: avoid; } table { break-inside: avoid; } }
"""


def generate_html_report(
    project_name: str,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    dataset: Dataset,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    final_result: FinalResult,
    cross_dataset_results: list[CrossDatasetValidationResult] | None = None,
    kfold_result: KFoldResult | None = None,
    repeated_kfold_result: RepeatedKFoldResult | None = None,
) -> str:
    """설계 문서 12번 형식의 자체완결형 HTML 리포트 문자열을 만든다."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    chosen_label = _MODEL_LABELS[final_result.chosen_model]

    grade = final_result.overall_grade
    star_count = _GRADE_STARS.get(grade, 0)
    stars = "\u2605" * star_count + "\u2606" * (5 - star_count)
    grade_color = _GRADE_COLOR.get(grade, "#666")
    grade_label = _GRADE_LABEL_EN.get(grade, grade.value)
    confidence_html = ""
    if final_result.confidence:
        conf = final_result.confidence
        comp_rows = "".join(
            f"<tr><th>{_esc(name)}</th><td>{value:.1f}/100</td></tr>"
            for name, value in conf.components.items()
        )
        reason_items = "".join(f"<li>{_esc(r)}</li>" for r in conf.reasons)
        warning_items = "".join(f"<li>{_esc(w)}</li>" for w in conf.warnings)
        confidence_html = (
            f"<h4>Final Calibration Confidence</h4>"
            f'<div class="grade-box" style="background:{grade_color};">{conf.score:.0f}/100 - {_esc(conf.level)}</div>'
            + (f'<table class="kv"><tbody>{comp_rows}</tbody></table>' if comp_rows else "")
            + (f"<ul>{reason_items}</ul>" if reason_items else "")
            + (f'<ul style="color:#ef6c00;">{warning_items}</ul>' if warning_items else "")
        )

    body = f"""
<h1>Camera Calibration Report</h1>
<div class="meta">Project: {_esc(project_name)} | Generated: {_esc(generated_at)}
| Chosen Model: <b>{_esc(chosen_label)}</b></div>

<h2>1. Camera &amp; Pattern</h2>
{_section_camera_pattern(camera_config, pattern_config, final_result.calibration)}

<h2>2. Dataset</h2>
{_section_dataset(dataset)}

<h2>3. Dataset Quality &amp; Detection Statistics</h2>
{_section_detection_statistics(dataset)}

<h2>4. Cross Validation</h2>
{_section_cross_validation(validation_results, kfold_result, repeated_kfold_result)}

<h2>5. Model Comparison</h2>
{_section_model_comparison(calibration_results, validation_results, final_result.model_scores, final_result.chosen_model)}

<h2>6. Chosen Model Detail - {_esc(chosen_label)}</h2>
{_section_chosen_model_detail(final_result.calibration, final_result.validation)}

<h2>7. Bootstrap Stability</h2>
{_section_bootstrap_stability(calibration_results)}

<h2>8. Sanity Check</h2>
{_section_sanity_check(camera_config, final_result.calibration)}

<h2>9. Outlier</h2>
{_section_outlier(final_result)}

<h2>Diagnosis &amp; Recommendations</h2>
{_section_diagnosis(final_result)}

<h2>Cross-Dataset Generalization</h2>
{_section_cross_dataset(cross_dataset_results)}

<h2>Final Calibration Summary</h2>
{_section_final_summary(final_result)}

<h2>Overall Quality</h2>
<div class="grade-box" style="background:{grade_color};">
  <span class="stars">{stars}</span>&nbsp; {_esc(grade_label)}
</div>
{confidence_html}
<p style="color:#777; font-size:12px; margin-top:10px;">
  This grade is a guideline based on the worst of Train/Test/Edge RMS and Line Straightness
  (design doc section 3.1) - not an absolute pass/fail threshold. The right tolerance depends
  on your sensor, lens, and application, so the final judgment is still yours.
</p>
"""
    return (
        "<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
        f"<title>{_esc(project_name)} - Calibration Report</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


def export_html_report(
    project_name: str,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    dataset: Dataset,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    final_result: FinalResult,
    path: str,
    cross_dataset_results: list[CrossDatasetValidationResult] | None = None,
    kfold_result: KFoldResult | None = None,
    repeated_kfold_result: RepeatedKFoldResult | None = None,
) -> str:
    html_str = generate_html_report(
        project_name, camera_config, pattern_config, dataset,
        calibration_results, validation_results, final_result, cross_dataset_results,
        kfold_result, repeated_kfold_result,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return path
