"""
camera_calibrator.ui.windshield_reflection_panel
==============================================================

Priority 7 안정화 - `ui/windshield_workspace.py`(God Object)에서 분리한
⑤ Reflection 탭의 Evaluation sub-tab 전용 UI. Reflection Suppression은
별도 파일(ui/windshield_reflection_suppression_panel.py)이 담당한다 -
Evaluation과 Suppression을 절대 섞지 않는다(사용자 스펙 원칙).

`ReflectionPanelMixin`은 `WindshieldWorkspace`에 다른 패널 mixin들과 함께
다중 상속되어 같은 `self`를 공유한다(자세한 설명은 ui/windshield_common.py
참고) - `_build_reflection_tab()`이 여기서
`self._build_reflection_suppression_subtab()`(다른 mixin에 정의됨)을
호출하지만, Python MRO가 어느 mixin에 정의됐든 찾아주므로 아무 문제 없다.

이 파일은 로직을 하나도 새로 추가/변경하지 않았다 - `ui/windshield_workspace.py`
에 있던 메서드를 그대로 옮겼을 뿐이다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.windshield.reflection import ReflectionDatasetResult, ReflectionEvaluationConfig, ReflectionImagePair
from export.reflection import export_reflection_yaml
from ui.reflection_worker import ReflectionEvaluationWorker
from ui.windshield_common import (
    ResponsiveRow,
    _ScrollTable,
    _fit_table_to_rows,
    _fmt,
    configure_form_layout,
    make_scrollable_page,
)
from ui.worker import run_worker_in_thread


class ReflectionPanelMixin:
    """⑤ Reflection 탭 컨테이너 + Evaluation sub-tab 전용 UI."""

    def _build_reflection_tab(self) -> QWidget:
        """STEP 7 - Evaluation과 Suppression을 별도 sub-tab으로 분리한다
        (사용자 스펙 0/49번, "Reflection Evaluation != Reflection
        Suppression"). 기존 Evaluation UI는 그대로 `_build_reflection_
        evaluation_subtab()`로 옮겼을 뿐 내용은 손대지 않았다."""
        outer = QTabWidget()
        outer.tabBar().setUsesScrollButtons(True)
        outer.tabBar().setExpanding(False)
        outer.setElideMode(Qt.ElideNone)
        outer.addTab(make_scrollable_page(self._build_reflection_evaluation_subtab()), "Evaluation")
        outer.addTab(make_scrollable_page(self._build_reflection_suppression_subtab()), "Suppression")
        return outer

    def _build_reflection_evaluation_subtab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        mode_group = QGroupBox("PHOTOMETRIC QUALITY")
        mode_layout = QVBoxLayout(mode_group)
        mode_row = QHBoxLayout()
        self.reflection_reference_radio = QRadioButton("Reference Pair")
        self.reflection_reference_radio.setChecked(True)
        self.reflection_no_reference_radio = QRadioButton("No-Reference")
        mode_row.addWidget(self.reflection_reference_radio)
        mode_row.addWidget(self.reflection_no_reference_radio)
        mode_row.addStretch(1)
        mode_layout.addLayout(mode_row)

        path_form = QFormLayout()
        self.reflection_normal_path_label = QLabel("N/A")
        self.reflection_reference_path_label = QLabel("N/A")
        normal_btn = QPushButton("Load Normal Image...")
        normal_btn.clicked.connect(self._on_load_reflection_normal_image)
        reference_btn = QPushButton("Load Reference Image...")
        reference_btn.clicked.connect(self._on_load_reflection_reference_image)
        normal_row = ResponsiveRow(breakpoint=620)
        normal_row.addWidget(normal_btn)
        normal_row.addWidget(self.reflection_normal_path_label, stretch=1)
        reference_row = ResponsiveRow(breakpoint=620)
        reference_row.addWidget(reference_btn)
        reference_row.addWidget(self.reflection_reference_path_label, stretch=1)
        path_form.addRow("Normal:", normal_row)
        path_form.addRow("Reference:", reference_row)

        self.reflection_threshold_spin = QDoubleSpinBox()
        self.reflection_threshold_spin.setRange(0.001, 1.0)
        self.reflection_threshold_spin.setDecimals(3)
        self.reflection_threshold_spin.setSingleStep(0.01)
        self.reflection_threshold_spin.setValue(0.08)
        path_form.addRow("Coverage Threshold:", self.reflection_threshold_spin)
        self.reflection_normal_path_label.setWordWrap(True)
        self.reflection_reference_path_label.setWordWrap(True)
        configure_form_layout(path_form)
        mode_layout.addLayout(path_form)

        action_row = ResponsiveRow(breakpoint=560)
        self.reflection_run_button = QPushButton("Run Evaluation")
        self.reflection_run_button.clicked.connect(self._on_run_reflection_evaluation)
        self.reflection_export_button = QPushButton("Export YAML...")
        self.reflection_export_button.setEnabled(False)
        self.reflection_export_button.clicked.connect(self._on_export_reflection_yaml)
        action_row.addWidget(self.reflection_run_button)
        action_row.addWidget(self.reflection_export_button)
        action_row.addStretch(1)
        mode_layout.addWidget(action_row)
        self.reflection_status_label = QLabel("Raw image-domain photometric evaluation. Geometry calibration is not modified.")
        self.reflection_status_label.setWordWrap(True)
        mode_layout.addWidget(self.reflection_status_label)
        layout.addWidget(mode_group)

        self.reflection_metrics_table = _ScrollTable(15, 1)
        self.reflection_metrics_table.setHorizontalHeaderLabels(["Value"])
        self.reflection_metrics_table.setVerticalHeaderLabels([
            "Mode", "Alignment", "Reflection Mean", "Reflection Median", "Reflection P95", "Reflection P99", "Reflection Coverage",
            "Reflection Likelihood",
            "Bottom Mean", "Bottom Coverage", "Contrast Retention",
            "Edge Retention", "Saturation Coverage", "Glare Coverage", "Glare Strength",
        ])
        layout.addWidget(self.reflection_metrics_table)

        self.reflection_spatial_table = _ScrollTable(4, 6)
        self.reflection_spatial_table.setHorizontalHeaderLabels([f"C{c+1}" for c in range(6)])
        self.reflection_spatial_table.setVerticalHeaderLabels([f"R{r+1}" for r in range(4)])
        layout.addWidget(self.reflection_spatial_table)
        layout.addStretch(1)
        return page

    def _on_load_reflection_normal_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Normal Reflection Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._reflection_normal_path = path
        self.reflection_normal_path_label.setText(path)

    def _on_load_reflection_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Reflection-Reduced Reference Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._reflection_reference_path = path
        self.reflection_reference_path_label.setText(path)

    def _on_run_reflection_evaluation(self) -> None:
        normal_path = getattr(self, "_reflection_normal_path", "")
        reference_path = getattr(self, "_reflection_reference_path", "")
        if not normal_path:
            QMessageBox.warning(self, "Reflection Evaluation", "Normal image is required.")
            return
        mode = "no_reference" if self.reflection_no_reference_radio.isChecked() else "reference"
        if mode == "reference" and not reference_path:
            QMessageBox.warning(self, "Reflection Evaluation", "Reference mode requires a reflection-reduced reference image.")
            return
        cfg = ReflectionEvaluationConfig(mode=mode, coverage_threshold=float(self.reflection_threshold_spin.value()))
        pair = ReflectionImagePair(normal_image_path=normal_path, reference_image_path=reference_path or None, pair_id="scene_001")
        worker = ReflectionEvaluationWorker([pair], cfg)
        thread = run_worker_in_thread(worker, self)
        self.reflection_run_button.setEnabled(False)
        self.reflection_status_label.setText("Reflection evaluation running...")
        worker.result_ready.connect(self._on_reflection_evaluation_finished)
        worker.error.connect(self._on_reflection_evaluation_error)
        worker.progress.connect(self.reflection_status_label.setText)
        self._reflection_thread, self._reflection_worker = thread, worker
        thread.start()

    def _on_reflection_evaluation_finished(self, result: ReflectionDatasetResult) -> None:
        self._reflection_result = result
        self._reflection_results["latest"] = result
        self._display_reflection_result(result)
        self.reflection_run_button.setEnabled(True)
        self.reflection_export_button.setEnabled(result.success)

    def _on_reflection_evaluation_error(self, message: str) -> None:
        self.reflection_status_label.setText(message)
        self.reflection_run_button.setEnabled(True)
        QMessageBox.critical(self, "Reflection Evaluation", message)

    def _display_reflection_result(self, dataset_result: ReflectionDatasetResult) -> None:
        result = dataset_result.pair_results[0] if dataset_result.pair_results else None
        if result is None:
            self.reflection_status_label.setText(dataset_result.error_message or "No reflection result.")
            return
        status_suffix = (
            "Reflection Likelihood: no-reference heuristic, not ground truth."
            if result.mode == "no_reference"
            else "Reference reflection evaluation after alignment and photometric normalization."
        )
        self.reflection_status_label.setText(f"Metric v{result.metric_version} | {status_suffix}")
        values = [
            result.mode,
            f"{result.alignment_status} ({_fmt(result.alignment_score)})",
            f"{result.reflection_mean * 100.0:.2f}%" if result.reflection_mean is not None else "N/A",
            f"{result.reflection_median * 100.0:.2f}%" if result.reflection_median is not None else "N/A",
            f"{result.reflection_p95 * 100.0:.2f}%" if result.reflection_p95 is not None else "N/A",
            f"{result.reflection_p99 * 100.0:.2f}%" if result.reflection_p99 is not None else "N/A",
            f"{result.reflection_coverage * 100.0:.2f}%" if result.reflection_coverage is not None else "N/A",
            f"{result.reflection_likelihood * 100.0:.2f}%" if result.reflection_likelihood is not None else "N/A",
            f"{(result.bottom_roi_mean_strength or 0.0) * 100.0:.2f}%",
            f"{(result.bottom_roi_coverage or 0.0) * 100.0:.2f}%",
            f"{result.contrast_retention * 100.0:.2f}%" if result.contrast_retention is not None else "N/A",
            f"{result.edge_retention * 100.0:.2f}%" if result.edge_retention is not None else "N/A",
            f"{result.saturation_coverage * 100.0:.2f}%",
            f"{(result.glare_coverage or 0.0) * 100.0:.2f}%",
            f"{(result.glare_strength or 0.0) * 100.0:.2f}%",
        ]
        for row, value in enumerate(values):
            self.reflection_metrics_table.setItem(row, 0, QTableWidgetItem(value))
        _fit_table_to_rows(self.reflection_metrics_table)

        for cell in result.spatial_map:
            if cell.row < self.reflection_spatial_table.rowCount() and cell.col < self.reflection_spatial_table.columnCount():
                self.reflection_spatial_table.setItem(cell.row, cell.col, QTableWidgetItem(f"{cell.mean_strength * 100.0:.1f}%"))
        _fit_table_to_rows(self.reflection_spatial_table)

    def _on_export_reflection_yaml(self) -> None:
        if self._reflection_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Reflection YAML 저장", "reflection_evaluation.yml", "YAML (*.yml *.yaml)")
        if not path:
            return
        try:
            export_reflection_yaml(self._reflection_result, path)
            self.reflection_status_label.setText(f"Reflection YAML saved: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Reflection Export", str(e))
