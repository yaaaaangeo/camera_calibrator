"""
camera_calibrator.ui.diagnosis_view
===================================

Diagnosis와 next capture recommendation을 보여주는 전용 탭.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from calibration.diagnosis import diagnose_calibration
from calibration.quality import coverage_percentage
from calibration.types import CalibrationResult, CameraModelType, Dataset, ValidationResult

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


class DiagnosisView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel("아직 diagnosis 결과가 없습니다.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.pattern_table = QTableWidget(0, 5)
        self.pattern_table.setHorizontalHeaderLabels(["Model", "Severity", "Pattern", "Evidence", "Recommendation"])
        self.pattern_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pattern_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.pattern_table, stretch=2)

        self.capture_table = QTableWidget(0, 5)
        self.capture_table.setHorizontalHeaderLabels(["Model", "Priority", "Next Capture", "Action", "Reason"])
        self.capture_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.capture_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.capture_table, stretch=1)

    def set_results(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        validation_results: dict[CameraModelType, ValidationResult],
        dataset: Dataset | None,
    ) -> None:
        self.pattern_table.setRowCount(0)
        self.capture_table.setRowCount(0)
        if not calibration_results:
            self.summary_label.setText("아직 diagnosis 결과가 없습니다.")
            return

        coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset and dataset.coverage_grid else None
        coverage_grid = dataset.coverage_grid if dataset else None
        diversity = dataset.diversity if dataset else None
        n_errors = n_warnings = n_captures = 0

        for model in _MODEL_ORDER:
            cal = calibration_results.get(model)
            if cal is None:
                continue
            report = diagnose_calibration(
                cal,
                validation_results.get(model),
                dataset_coverage_pct=coverage_pct,
                coverage_grid=coverage_grid,
                diversity=diversity,
            )
            for pattern in report.patterns:
                row = self.pattern_table.rowCount()
                self.pattern_table.insertRow(row)
                evidence = "\n".join(pattern.evidence) if pattern.evidence else "-"
                values = [
                    _MODEL_LABELS.get(model, model.value),
                    pattern.severity.value.upper(),
                    f"{pattern.title}\n{pattern.code}",
                    evidence,
                    pattern.recommendation or "-",
                ]
                for col, value in enumerate(values):
                    self.pattern_table.setItem(row, col, QTableWidgetItem(value))
                if pattern.severity.value == "error":
                    n_errors += 1
                elif pattern.severity.value == "warning":
                    n_warnings += 1

            for rec in report.capture_recommendations:
                row = self.capture_table.rowCount()
                self.capture_table.insertRow(row)
                values = [
                    _MODEL_LABELS.get(model, model.value),
                    rec.priority.upper(),
                    f"{rec.title}\n{rec.code}",
                    rec.action,
                    rec.reason or "-",
                ]
                for col, value in enumerate(values):
                    self.capture_table.setItem(row, col, QTableWidgetItem(value))
                n_captures += 1

        self.summary_label.setText(
            f"Diagnosis patterns: errors {n_errors}, warnings {n_warnings}. "
            f"Next capture recommendations: {n_captures}."
        )
