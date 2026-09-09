"""
camera_calibrator.ui.stability_view
===================================

Parameter stability, observability, undistortion quality를 한 화면에 모은 탭.

Paper Evidence 단계 추가 - 기본 표의 "Param Stability" 열은 여전히
ParameterUncertainty.overall_stability(all-parameter, recommender.py가
실제로 쓰는 legacy 값)를 그대로 보여준다(값도 의미도 바꾸지 않았다).
"왜 Fisheye가 62%인지"처럼 그 숫자의 근거를 보고 싶을 때를 위해, 표에서
모델 행을 선택하면 아래 detail 패널에 method(어떤 실험에서 나온 숫자인지 -
bootstrap frame resampling인지 covariance 근사인지), fx/fy/cx/cy 각각의
reference/mean/std/95% CI/stability, distortion coefficient별 같은 정보,
Paper Intrinsic Stability(fx/fy/cx/cy만)와 All-Parameter Stability를
나란히, 그리고 가장 낮은 stability를 가진 파라미터와 near-zero reference
경고를 보여준다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.recommender import compute_final_result
from calibration.quality import coverage_percentage
from calibration.types import (
    CalibrationResult,
    CameraModelType,
    Dataset,
    ModelScore,
    ParameterUncertainty,
    ValidationResult,
)

_METHOD_LABELS = {
    "bootstrap": "Bootstrap Parameter Stability (frame resampling with replacement)",
    "covariance": "Covariance-based (cv2 stdDeviationsIntrinsics, normal approximation)",
}

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [
    CameraModelType.PINHOLE,
    CameraModelType.BROWN_CONRADY,
    CameraModelType.EXTENDED_PINHOLE,
    CameraModelType.FISHEYE,
]


def _fmt(v: float | None, suffix: str = "") -> str:
    return f"{v:.3f}{suffix}" if v is not None else "N/A"


def _fmt_score(v: float | None) -> str:
    return f"{v:.1f}/100" if v is not None else "N/A"


def _fmt_param_detail(name: str, reference: float | None, mean: float | None, std: float | None,
                       ci_low: float | None, ci_high: float | None, stability: float | None) -> str:
    if mean is None and std is None:
        return f"  {name}: N/A"
    ref_s = f"{reference:.3f}" if reference is not None else "N/A"
    mean_s = f"{mean:.3f}" if mean is not None else "N/A"
    std_s = f"{std:.3f}" if std is not None else "N/A"
    ci_s = f"{ci_low:.1f} ~ {ci_high:.1f}" if ci_low is not None and ci_high is not None else "N/A"
    stab_s = _fmt_score(stability)
    return f"  {name}: reference={ref_s}, mean={mean_s}, std={std_s}, 95% CI=[{ci_s}], stability={stab_s}"


def format_stability_detail(model_label: str, pu: ParameterUncertainty | None) -> str:
    """모델 하나를 선택했을 때 보여줄 상세 텍스트 - Paper Evidence 단계 추가.

    기존 표의 "Param Stability" 열(=overall_stability, recommender.py가
    실제로 쓰는 legacy all-parameter 값)은 여기서도 그대로 다시 보여준다 -
    이 함수는 그 숫자의 근거(어느 파라미터가 stability를 끌어내렸는지)를
    추가로 설명할 뿐, 값을 재계산하거나 다르게 만들지 않는다.
    """
    if pu is None:
        return f"{model_label}: Parameter Uncertainty가 계산되지 않았습니다."

    lines = [f"{model_label}"]
    method_label = _METHOD_LABELS.get(pu.method, pu.method)
    lines.append(f"Method: {method_label}")
    if pu.method == "bootstrap":
        total = pu.n_bootstrap_total if pu.n_bootstrap_total is not None else "N/A"
        success = pu.n_bootstrap_success if pu.n_bootstrap_success is not None else "N/A"
        lines.append(f"Bootstrap successful samples / total samples: {success} / {total}")

    lines.append(_fmt_param_detail("fx", pu.fx_reference, pu.fx_mean, pu.fx_std, pu.fx_ci_low, pu.fx_ci_high, pu.fx_stability))
    lines.append(_fmt_param_detail("fy", pu.fy_reference, pu.fy_mean, pu.fy_std, pu.fy_ci_low, pu.fy_ci_high, pu.fy_stability))
    lines.append(_fmt_param_detail("cx", pu.cx_reference, pu.cx_mean, pu.cx_std, pu.cx_ci_low, pu.cx_ci_high, pu.cx_stability))
    lines.append(_fmt_param_detail("cy", pu.cy_reference, pu.cy_mean, pu.cy_std, pu.cy_ci_low, pu.cy_ci_high, pu.cy_stability))

    if pu.distortion_stats:
        lines.append("Distortion coefficients:")
        for stat in pu.distortion_stats:
            label = stat.label or f"d{stat.index}"
            line = _fmt_param_detail(label, stat.reference, stat.mean, stat.std, stat.ci_low, stat.ci_high, stat.stability_score)
            if stat.diagnostic:
                line += f"  [{stat.diagnostic}]"
            lines.append(line)

    lines.append(
        f"Paper Intrinsic Stability (fx/fy/cx/cy only) = {_fmt_score(pu.paper_intrinsic_stability)}"
    )
    lines.append(f"All-Parameter Stability (legacy overall_stability) = {_fmt_score(pu.overall_stability)}")
    if pu.distortion_stability_summary is not None:
        lines.append(f"Distortion Stability Summary (diagnostic only) = {_fmt_score(pu.distortion_stability_summary)}")
    if pu.lowest_stability_parameter is not None:
        lines.append(
            f"Lowest-stability parameter: {pu.lowest_stability_parameter} "
            f"({_fmt_score(pu.lowest_stability_value)})"
        )
    near_zero = [
        (s.label or f"d{s.index}") for s in pu.distortion_stats if s.near_zero_reference
    ]
    if near_zero:
        lines.append(
            f"near-zero CV warning: {', '.join(near_zero)} - reference값이 0에 가까워 "
            f"relative CV(및 그로부터 나온 stability)가 통계적으로 불안정할 수 있습니다."
        )
    return "\n".join(lines)


class StabilityView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel("아직 stability 결과가 없습니다.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 13)
        self.table.setHorizontalHeaderLabels([
            "Model", "fx std", "fy std", "cx std", "cy std", "Param Stability",
            "Bootstrap N", "Observability", "Norm Condition", "Raw Condition", "Max Corr",
            "Undistortion", "Final Confidence",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        # --- 모델 선택 시 상세 정보(Paper Evidence 단계 추가) ---
        self._pu_by_row: dict[int, ParameterUncertainty | None] = {}
        self._model_label_by_row: dict[int, str] = {}
        self.detail_label = QLabel("모델 행을 선택하면 fx/fy/cx/cy/distortion coefficient별 상세 정보를 볼 수 있습니다.")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(self.detail_label.textInteractionFlags() | Qt.TextSelectableByMouse)
        layout.addWidget(self.detail_label)
        self.table.itemSelectionChanged.connect(self._on_row_selected)

    def _on_row_selected(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if not rows:
            return
        row = next(iter(rows))
        pu = self._pu_by_row.get(row)
        label = self._model_label_by_row.get(row, "")
        self.detail_label.setText(format_stability_detail(label, pu))

    def set_results(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        validation_results: dict[CameraModelType, ValidationResult],
        scores: list[ModelScore],
        dataset: Dataset | None,
    ) -> None:
        self.table.setRowCount(0)
        self._pu_by_row = {}
        self._model_label_by_row = {}
        self.detail_label.setText("모델 행을 선택하면 fx/fy/cx/cy/distortion coefficient별 상세 정보를 볼 수 있습니다.")
        if not calibration_results:
            self.summary_label.setText("아직 stability 결과가 없습니다.")
            return

        coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset and dataset.coverage_grid else None
        confidence_values: list[float] = []

        for model in _MODEL_ORDER:
            cal = calibration_results.get(model)
            if cal is None:
                continue
            val = validation_results.get(model)
            pu = cal.param_uncertainty_bootstrap or cal.param_uncertainty
            obs = cal.observability
            uq = cal.undistortion_quality
            final = compute_final_result(
                model,
                calibration_results,
                validation_results,
                dataset_coverage_pct=coverage_pct,
                scores=scores,
                coverage_grid=dataset.coverage_grid if dataset else None,
                dataset_diversity=dataset.diversity if dataset else None,
            )
            if final.confidence:
                confidence_values.append(final.confidence.score)

            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                _MODEL_LABELS.get(model, model.value),
                _fmt(pu.fx_std if pu else None),
                _fmt(pu.fy_std if pu else None),
                _fmt(pu.cx_std if pu else None),
                _fmt(pu.cy_std if pu else None),
                _fmt_score(pu.overall_stability if pu else None),
                str(pu.n_bootstrap_success) if pu and pu.n_bootstrap_success is not None else "N/A",
                (
                    f"{obs.observability_grade or 'N/A'} {_fmt_score(obs.observability_score)}"
                    if obs else "N/A"
                ),
                (
                    f"{obs.normalized_condition_number:.3g}"
                    if obs and obs.normalized_condition_number is not None else "N/A"
                ),
                f"{obs.raw_condition_number:.3g}" if obs and obs.raw_condition_number is not None else "N/A",
                f"{obs.max_abs_correlation:.3f}" if obs and obs.max_abs_correlation is not None else "N/A",
                f"{uq.quality_grade.value.upper()} {_fmt_score(uq.quality_score)}" if uq else "N/A",
                (
                    f"{final.confidence.score:.0f}/100 {final.confidence.level}"
                    if final.confidence else "N/A"
                ),
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
            self._pu_by_row[row] = pu
            self._model_label_by_row[row] = _MODEL_LABELS.get(model, model.value)

        if confidence_values:
            self.summary_label.setText(
                f"Final confidence range: {min(confidence_values):.0f}/100 - {max(confidence_values):.0f}/100."
            )
        else:
            self.summary_label.setText("Stability metrics are not available yet.")
