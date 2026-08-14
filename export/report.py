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
    Dataset,
    FinalResult,
    ModelScore,
    PatternConfig,
    QualityGrade,
    ValidationResult,
)
from calibration.models.common import regional_edge_average
from calibration.quality import coverage_percentage

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
    return f"<p>{_esc(summary)}</p>{bars_html}"


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

    train_vals, test_vals, edge_vals, straight_vals, score_vals, chosen_vals = [], [], [], [], [], []
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
        score_vals.append(f"{score.score:.3f}" if score else "N/A")

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
        + row("Model Score", score_vals)
        + row("", chosen_vals)
        + "</tbody></table>"
    )
    return table


def _section_chosen_model_detail(cal: CalibrationResult | None, val: ValidationResult | None) -> str:
    if cal is None or not cal.success:
        return "<p>선택된 모델의 캘리브레이션 결과가 없습니다.</p>"

    K = cal.camera_matrix
    D = cal.distortion.ravel() if cal.distortion is not None else np.array([])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    rows = [
        ("fx, fy", f"{fx:.2f}, {fy:.2f}"),
        ("cx, cy", f"{cx:.2f}, {cy:.2f}"),
        ("Distortion coeffs", ", ".join(f"{v:.6f}" for v in D.tolist())),
        ("RMS Reprojection Error", _fmt(cal.rms_error)),
    ]
    if cal.param_uncertainty and cal.param_uncertainty.fx_std is not None:
        pu = cal.param_uncertainty
        # Fisheye는 OpenCV가 covariance를 안 줘서 bootstrap resampling으로 추정한
        # 값이다 (calibration/models/fisheye.py 참고) - Pinhole/Extended의
        # calibrateCameraExtended() 기반 표준편차와 계산 방식이 다르므로,
        # 리포트를 보는 사람이 "왜 숫자가 다르게 느껴지지" 헷갈리지 않도록 표시한다.
        method_note = " (bootstrap 추정)" if cal.model_name == CameraModelType.FISHEYE else ""
        rows.append((f"fx/fy std dev{method_note}", f"{pu.fx_std:.3f} / {pu.fy_std:.3f}"))
        within = pu.is_within_threshold(fx, fy)
        rows.append(("Parameter Stability", "Good (within 1%)" if within else "Warning (over 1%)"))

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

    return detail


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
) -> str:
    """설계 문서 12번 형식의 자체완결형 HTML 리포트 문자열을 만든다."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    chosen_label = _MODEL_LABELS[final_result.chosen_model]

    grade = final_result.overall_grade
    star_count = _GRADE_STARS.get(grade, 0)
    stars = "\u2605" * star_count + "\u2606" * (5 - star_count)
    grade_color = _GRADE_COLOR.get(grade, "#666")
    grade_label = _GRADE_LABEL_EN.get(grade, grade.value)

    body = f"""
<h1>Camera Calibration Report</h1>
<div class="meta">Project: {_esc(project_name)} | Generated: {_esc(generated_at)}
| Chosen Model: <b>{_esc(chosen_label)}</b></div>

<h2>1. Camera &amp; Pattern</h2>
{_section_camera_pattern(camera_config, pattern_config, final_result.calibration)}

<h2>2. Dataset</h2>
{_section_dataset(dataset)}

<h2>3. Model Comparison</h2>
{_section_model_comparison(calibration_results, validation_results, final_result.model_scores, final_result.chosen_model)}

<h2>4. Chosen Model Detail - {_esc(chosen_label)}</h2>
{_section_chosen_model_detail(final_result.calibration, final_result.validation)}

<h2>5. Outlier</h2>
{_section_outlier(final_result)}

<h2>6. Overall Quality</h2>
<div class="grade-box" style="background:{grade_color};">
  <span class="stars">{stars}</span>&nbsp; {_esc(grade_label)}
</div>
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
) -> str:
    html_str = generate_html_report(
        project_name, camera_config, pattern_config, dataset,
        calibration_results, validation_results, final_result,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return path
