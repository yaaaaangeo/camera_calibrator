"""Presentation-only view for leak-safe calibration refinement."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from calibration.error_normalization import normalized_reprojection_error
from calibration.models.common import distortion_coeff_labels
from calibration.optimizer import PIPELINE_STEPS
from calibration.types import CameraModelType, OptimizerResult, OptimizerSettings


MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}
STATUS_ICON = {"pending": "○", "running": "◐", "completed": "✓", "failed": "✕"}


class OptimizerView(QWidget):
    run_requested = Signal(object, object)       # model, settings
    cancel_requested = Signal()
    apply_requested = Signal(object)
    restore_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calibrations = {}; self._validations = {}; self._results = {}; self._camera = None
        root = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        page = QWidget(); self.page_layout = QVBoxLayout(page)
        scroll.setWidget(page); root.addWidget(scroll)

        current = QGroupBox("Current Calibration")
        form = QFormLayout(current)
        self.model_combo = QComboBox()
        for model, label in MODEL_LABELS.items(): self.model_combo.addItem(label, model)
        self.model_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("Model", self.model_combo)
        self.resolution_label = QLabel("N/A"); form.addRow("Resolution", self.resolution_label)
        self.frames_label = QLabel("Train 0 / Hold-out 0"); form.addRow("Frames", self.frames_label)
        self.parameters_label = QLabel("N/A"); self.parameters_label.setWordWrap(True)
        form.addRow("Current Parameters", self.parameters_label)
        self.page_layout.addWidget(current)

        settings_box = QGroupBox("Optimizer Settings")
        settings_form = QFormLayout(settings_box)
        self.multi = QCheckBox(); self.multi.setChecked(True); settings_form.addRow("Enable Multi-start", self.multi)
        self.starts = QSpinBox(); self.starts.setRange(1, 8); self.starts.setValue(4); settings_form.addRow("Number of Starts", self.starts)
        self.loss = QComboBox(); self.loss.addItems(["Huber", "Linear", "Soft-L1", "Cauchy"]); settings_form.addRow("Robust Loss", self.loss)
        self.delta = QDoubleSpinBox(); self.delta.setRange(0.01, 100.0); self.delta.setValue(1.0); settings_form.addRow("Huber Delta (px)", self.delta)
        self.staged = QCheckBox(); self.staged.setChecked(True); settings_form.addRow("Enable Staged Optimization", self.staged)
        self.stage_iterations = QSpinBox(); self.stage_iterations.setRange(1, 2000); self.stage_iterations.setValue(100); settings_form.addRow("Max Iterations / Stage", self.stage_iterations)
        self.final_iterations = QSpinBox(); self.final_iterations.setRange(1, 5000); self.final_iterations.setValue(200); settings_form.addRow("Final Joint Iterations", self.final_iterations)
        self.opt_focal = QCheckBox(); self.opt_focal.setChecked(True); settings_form.addRow("Optimize Focal Length", self.opt_focal)
        self.opt_pp = QCheckBox(); self.opt_pp.setChecked(True); settings_form.addRow("Optimize Principal Point", self.opt_pp)
        self.opt_dist = QCheckBox(); self.opt_dist.setChecked(True); settings_form.addRow("Optimize Distortion", self.opt_dist)
        self.opt_ext = QCheckBox(); self.opt_ext.setChecked(True); settings_form.addRow("Optimize Extrinsics", self.opt_ext)
        self.page_layout.addWidget(settings_box)

        pipeline_box = QGroupBox("Optimizer Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_box)
        self.pipeline_table = QTableWidget(len(PIPELINE_STEPS), 2)
        self.pipeline_table.setHorizontalHeaderLabels(["Step", "Status"])
        for row, step in enumerate(PIPELINE_STEPS):
            self.pipeline_table.setItem(row, 0, QTableWidgetItem(step))
            self.pipeline_table.setItem(row, 1, QTableWidgetItem("○ Pending"))
        pipeline_layout.addWidget(self.pipeline_table); self.page_layout.addWidget(pipeline_box)

        compare_box = QGroupBox("Before / After (same frozen hold-out)")
        compare_layout = QVBoxLayout(compare_box)
        self.compare_table = QTableWidget(7, 4)
        self.compare_table.setHorizontalHeaderLabels(["Metric", "Before", "After", "Delta"])
        for row, name in enumerate(("Train RMS", "Test RMS", "Test P95", "Test P99", "Edge RMS", "Stability", "Observability")):
            self.compare_table.setItem(row, 0, QTableWidgetItem(name))
        compare_layout.addWidget(self.compare_table)
        self.verdict = QLabel("Run the optimizer to compare results."); self.verdict.setWordWrap(True)
        compare_layout.addWidget(self.verdict); self.page_layout.addWidget(compare_box)

        details_box = QGroupBox("Optimization Details")
        details_layout = QVBoxLayout(details_box)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMinimumHeight(180)
        details_layout.addWidget(self.details); self.page_layout.addWidget(details_box)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run Optimizer"); self.run_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.setEnabled(False)
        self.apply_button = QPushButton("Apply Optimized Calibration"); self.apply_button.setEnabled(False)
        self.restore_button = QPushButton("Restore Original"); self.restore_button.setEnabled(False)
        self.run_button.clicked.connect(self._emit_run); self.cancel_button.clicked.connect(self.cancel_requested)
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.current_model()))
        self.restore_button.clicked.connect(lambda: self.restore_requested.emit(self.current_model()))
        for button in (self.run_button, self.cancel_button, self.apply_button, self.restore_button): buttons.addWidget(button)
        self.page_layout.addLayout(buttons); self.page_layout.addStretch(1)

    def current_model(self):
        return self.model_combo.currentData()

    def settings(self):
        return OptimizerSettings(
            multi_start=self.multi.isChecked(), num_starts=self.starts.value(),
            robust_loss=self.loss.currentText().lower().replace("-", "_"), huber_delta=self.delta.value(),
            staged=self.staged.isChecked(), max_iterations_per_stage=self.stage_iterations.value(),
            final_joint_iterations=self.final_iterations.value(), optimize_principal_point=self.opt_pp.isChecked(),
            optimize_focal_length=self.opt_focal.isChecked(), optimize_distortion=self.opt_dist.isChecked(),
            optimize_extrinsics=self.opt_ext.isChecked(),
        )

    def set_context(self, calibrations, validations, results, camera_config):
        self._calibrations = calibrations or {}; self._validations = validations or {}
        self._results = results or {}; self._camera = camera_config; self._refresh()

    def select_model(self, model):
        index = self.model_combo.findData(model)
        if index >= 0: self.model_combo.setCurrentIndex(index)

    def set_running(self, running: bool):
        self.run_button.setEnabled(not running and self._can_run())
        self.cancel_button.setEnabled(running)
        if running:
            self.apply_button.setEnabled(False); self.restore_button.setEnabled(False)
            for row in range(len(PIPELINE_STEPS)):
                self.pipeline_table.setItem(row, 1, QTableWidgetItem("○ Pending"))
            self.pipeline_table.setItem(0, 1, QTableWidgetItem("◐ Running"))

    def set_failed(self, message: str):
        for row in range(len(PIPELINE_STEPS)):
            text = self.pipeline_table.item(row, 1).text()
            if "Running" in text:
                self.pipeline_table.setItem(row, 1, QTableWidgetItem("✕ Failed"))
                break
        self.details.setPlainText(message)

    def set_cancelled(self):
        self.details.setPlainText("Cancelled. Original OpenCV calibration was preserved.")

    def _can_run(self):
        cal = self._calibrations.get(self.current_model()); val = self._validations.get(self.current_model())
        return bool(cal and cal.success and val and val.train_frame_ids)

    def _emit_run(self):
        self.run_requested.emit(self.current_model(), self.settings())

    @staticmethod
    def _fmt(value, cal=None, percent=False):
        if value is None: return "N/A"
        if percent: return f"{value:.1f}%"
        normalized = normalized_reprojection_error(value, cal.camera_matrix if cal else None)
        return f"{value:.3f} px\nNormalized: {normalized:.6f}" if normalized is not None else f"{value:.3f}"

    def _refresh(self):
        model = self.current_model(); cal = self._calibrations.get(model); val = self._validations.get(model)
        result: OptimizerResult | None = self._results.get(model)
        self.resolution_label.setText(f"{self._camera.width} x {self._camera.height}" if self._camera else "N/A")
        self.frames_label.setText(f"Train {len(val.train_frame_ids) if val else 0} / Hold-out {len(val.test_frame_ids) if val else 0}")
        if cal and cal.camera_matrix is not None:
            K = cal.camera_matrix; values = [f"fx={K[0,0]:.6g}", f"fy={K[1,1]:.6g}", f"cx={K[0,2]:.6g}", f"cy={K[1,2]:.6g}"]
            if cal.distortion is not None:
                values += [f"{n}={v:.6g}" for n, v in zip(distortion_coeff_labels(model, cal.distortion.size), cal.distortion.ravel())]
            self.parameters_label.setText("  ·  ".join(values))
        else: self.parameters_label.setText("N/A")
        self.run_button.setEnabled(self._can_run())
        if result:
            self._show_result(result)
        else:
            for row in range(7):
                for col in range(1, 4): self.compare_table.setItem(row, col, QTableWidgetItem("N/A"))
            self.apply_button.setEnabled(False); self.restore_button.setEnabled(False)

    def _show_result(self, result: OptimizerResult):
        for row, step in enumerate(PIPELINE_STEPS):
            state = result.pipeline_status.get(step, "pending")
            self.pipeline_table.setItem(row, 1, QTableWidgetItem(f"{STATUS_ICON.get(state, '○')} {state.title()}"))
        before = result.before_metrics; after = result.after_metrics
        attrs = ("train_rms", "test_rms", "test_p95", "test_p99", "edge_rms", "stability", "observability")
        for row, attr in enumerate(attrs):
            b = getattr(before, attr); a = getattr(after, attr); percent = row >= 5
            self.compare_table.setItem(row, 1, QTableWidgetItem(self._fmt(b, result.original_calibration, percent)))
            self.compare_table.setItem(row, 2, QTableWidgetItem(self._fmt(a, result.optimized_calibration, percent)))
            delta = None if b is None or a is None else a - b
            self.compare_table.setItem(row, 3, QTableWidgetItem("N/A" if delta is None else f"{delta:+.3f}"))
        self.verdict.setText(result.recommendation + "\n" + "\n".join(f"• {r}" for r in result.reasons))
        lines = ["Multi-start"]
        for start in result.starts:
            lines.append(f"  {start.name}: {'Converged' if start.converged else 'Failed'}  objective={start.objective} {'[Selected]' if start.selected else ''}\n    {start.message}")
        lines.append("\nStages")
        for stage in result.stages:
            lines.append(f"  {stage.name}: iterations={stage.iterations}, RMS {stage.rms_before:.6f} -> {stage.rms_after:.6f}\n    {', '.join(stage.active_parameters)}\n    {stage.termination}")
        if result.excluded_training_frame_ids:
            lines.append(f"\nWarning: training observations differ after Fisheye fallback; excluded: {', '.join(result.excluded_training_frame_ids)}")
        self.details.setPlainText("\n".join(lines))
        self.apply_button.setEnabled(result.success and not result.applied)
        self.restore_button.setEnabled(result.success and result.applied)
