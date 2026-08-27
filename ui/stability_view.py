"""
camera_calibrator.ui.stability_view
===================================

Parameter stability, observability, undistortion quality를 한 화면에 모은 탭.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from calibration.recommender import compute_final_result
from calibration.quality import coverage_percentage
from calibration.types import CalibrationResult, CameraModelType, Dataset, ModelScore, ValidationResult

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]


def _fmt(v: float | None, suffix: str = "") -> str:
    return f"{v:.3f}{suffix}" if v is not None else "N/A"


def _fmt_score(v: float | None) -> str:
    return f"{v:.1f}/100" if v is not None else "N/A"


class StabilityView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel("아직 stability 결과가 없습니다.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels([
            "Model", "fx std", "fy std", "cx std", "cy std", "Param Stability",
            "Bootstrap N", "Observability", "Condition", "Max Corr",
            "Undistortion", "Final Confidence",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

    def set_results(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        validation_results: dict[CameraModelType, ValidationResult],
        scores: list[ModelScore],
        dataset: Dataset | None,
    ) -> None:
        self.table.setRowCount(0)
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
                f"{obs.condition_number:.3g}" if obs and obs.condition_number is not None else "N/A",
                f"{obs.max_abs_correlation:.3f}" if obs and obs.max_abs_correlation is not None else "N/A",
                f"{uq.quality_grade.value.upper()} {_fmt_score(uq.quality_score)}" if uq else "N/A",
                (
                    f"{final.confidence.score:.0f}/100 {final.confidence.level}"
                    if final.confidence else "N/A"
                ),
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

        if confidence_values:
            self.summary_label.setText(
                f"Final confidence range: {min(confidence_values):.0f}/100 - {max(confidence_values):.0f}/100."
            )
        else:
            self.summary_label.setText("Stability metrics are not available yet.")
