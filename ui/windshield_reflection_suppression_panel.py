"""
camera_calibrator.ui.windshield_reflection_suppression_panel
==============================================================

Priority 7 안정화 - `ui/windshield_workspace.py`(God Object)에서 분리한
⑤ Reflection 탭의 Suppression sub-tab 전용 UI(사용자 스펙 0/49/57/58번,
"Reflection Evaluation != Reflection Suppression" - 별도 파일/모델/worker).

`ReflectionSuppressionPanelMixin`은 `WindshieldWorkspace`에 다른 패널
mixin들과 함께 다중 상속되어 같은 `self`를 공유한다(ui/windshield_common.py
참고).

이 파일은 로직을 하나도 새로 추가/변경하지 않았다 - `ui/windshield_workspace.py`
에 있던 메서드를 그대로 옮겼을 뿐이다. GUI는 여전히 학습(training loop)을
직접 돌리지 않는다 - `ui/reflection_suppression_worker.py`가 이미 학습된
모델로 inference만 수행한다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.windshield_common import _ScrollTable
from ui.worker import run_worker_in_thread


class ReflectionSuppressionPanelMixin:
    """⑤ Reflection 탭의 Suppression sub-tab 전용 UI."""

    def _build_reflection_suppression_subtab(self) -> QWidget:
        """STEP 7 - Reflection Suppression(사용자 스펙 50/51/52번). Training은
        여기 없다(사용자 스펙 59번) - 이미 학습된 model.yml을 불러와
        inference/시각화/Before-After 평가만 한다."""
        page = QWidget()
        layout = QVBoxLayout(page)

        self._suppression_model_path = ""
        self._suppression_input_path = ""
        self._suppression_result = None

        group = QGroupBox("REFLECTION SUPPRESSION")
        form = QFormLayout(group)

        model_row = QHBoxLayout()
        self.suppression_model_path_label = QLabel("N/A")
        load_model_btn = QPushButton("Load Model...")
        load_model_btn.clicked.connect(self._on_load_suppression_model)
        model_row.addWidget(load_model_btn)
        model_row.addWidget(self.suppression_model_path_label, stretch=1)
        form.addRow("Model:", model_row)

        input_row = QHBoxLayout()
        self.suppression_input_path_label = QLabel("N/A")
        load_input_btn = QPushButton("Load Image...")
        load_input_btn.clicked.connect(self._on_load_suppression_input_image)
        input_row.addWidget(load_input_btn)
        input_row.addWidget(self.suppression_input_path_label, stretch=1)
        form.addRow("Input:", input_row)

        # Reference는 선택 사항이다(사용자 스펙 53번, No-Reference Runtime) -
        # 있으면 STEP 6 evaluator로 Before/After를 Reference Mode로, 없으면
        # No-Reference Mode(Reflection Likelihood)로 평가한다.
        reference_row = QHBoxLayout()
        self._suppression_reference_path = ""
        self.suppression_reference_path_label = QLabel("N/A (No-Reference mode)")
        load_reference_btn = QPushButton("Load Reference (optional)...")
        load_reference_btn.clicked.connect(self._on_load_suppression_reference_image)
        reference_row.addWidget(load_reference_btn)
        reference_row.addWidget(self.suppression_reference_path_label, stretch=1)
        form.addRow("Reference:", reference_row)

        mode_row = QHBoxLayout()
        self._suppression_mode_button_group = QButtonGroup(self)
        self.suppression_mode_conservative_radio = QRadioButton("Conservative")
        self.suppression_mode_standard_radio = QRadioButton("Standard")
        self.suppression_mode_standard_radio.setChecked(True)
        self.suppression_mode_strong_radio = QRadioButton("Strong")
        for radio in (
            self.suppression_mode_conservative_radio,
            self.suppression_mode_standard_radio,
            self.suppression_mode_strong_radio,
        ):
            self._suppression_mode_button_group.addButton(radio)
            mode_row.addWidget(radio)
        mode_row.addStretch(1)
        form.addRow("Mode:", mode_row)
        layout.addWidget(group)

        action_row = QHBoxLayout()
        self.suppression_run_button = QPushButton("Run Suppression")
        self.suppression_run_button.clicked.connect(self._on_run_reflection_suppression)
        action_row.addWidget(self.suppression_run_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.suppression_status_label = QLabel(
            "Learned residual reflection-layer correction. Geometry(K,D/Spherical/Grid/RBF/"
            "Spline/Neural) is never modified by suppression."
        )
        self.suppression_status_label.setWordWrap(True)
        layout.addWidget(self.suppression_status_label)

        # Visualization: Original / Predicted Reflection / Reflection Mask / Suppressed
        # (사용자 스펙 51번, "단순 시각 효과가 아니라 디버깅에 중요하다").
        viz_group = QGroupBox("VISUALIZATION")
        viz_grid = QHBoxLayout(viz_group)
        self.suppression_original_image_label = QLabel("Original")
        self.suppression_reflection_image_label = QLabel("Predicted Reflection")
        self.suppression_alpha_image_label = QLabel("Reflection Mask")
        self.suppression_output_image_label = QLabel("Suppressed")
        for lbl in (
            self.suppression_original_image_label,
            self.suppression_reflection_image_label,
            self.suppression_alpha_image_label,
            self.suppression_output_image_label,
        ):
            lbl.setMinimumSize(160, 120)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid gray;")
            viz_grid.addWidget(lbl)
        layout.addWidget(viz_group)

        # Before/After metrics(사용자 스펙 52번) - STEP 6 evaluator를
        # suppression 전/후에 그대로 적용한 결과를 나란히 보여준다.
        self.suppression_metrics_table = _ScrollTable(6, 2)
        self.suppression_metrics_table.setHorizontalHeaderLabels(["Before", "After"])
        self.suppression_metrics_table.setVerticalHeaderLabels([
            "Reflection Mean", "Reflection P95", "Reflection Coverage",
            "Edge Retention", "Contrast Retention", "Over-suppression",
        ])
        layout.addWidget(self.suppression_metrics_table)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # STEP 8 - Ghost / Double Image. Reflection과 완전히 독립된 탭이다
    # (사용자 스펙 "Ghost는 기존 Reflection 기능 안에 넣지 않는다") - worker/
    # 상태/결과 타입 모두 별도이고, Displacement(vector field)와
    # Strength(heatmap)도 하나의 표에 억지로 합치지 않는다.
    # ------------------------------------------------------------------

    def _suppression_strength_value(self) -> float:
        from calibration.windshield.reflection_suppression.config import (
            SUPPRESSION_STRENGTH_CONSERVATIVE,
            SUPPRESSION_STRENGTH_STANDARD,
            SUPPRESSION_STRENGTH_STRONG,
        )

        if self.suppression_mode_conservative_radio.isChecked():
            return SUPPRESSION_STRENGTH_CONSERVATIVE
        if self.suppression_mode_strong_radio.isChecked():
            return SUPPRESSION_STRENGTH_STRONG
        return SUPPRESSION_STRENGTH_STANDARD

    def _on_load_suppression_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Reflection Suppression Model", "", "YAML (*.yml *.yaml)")
        if not path:
            return
        self._suppression_model_path = path
        self.suppression_model_path_label.setText(path)

    def _on_load_suppression_input_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Input Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._suppression_input_path = path
        self.suppression_input_path_label.setText(path)

    def _on_load_suppression_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Reflection-Reduced Reference Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._suppression_reference_path = path
        self.suppression_reference_path_label.setText(path)

    def _on_run_reflection_suppression(self) -> None:
        import cv2

        if not self._suppression_model_path:
            QMessageBox.warning(self, "Reflection Suppression", "Model이 필요합니다.")
            return
        if not self._suppression_input_path:
            QMessageBox.warning(self, "Reflection Suppression", "Input image가 필요합니다.")
            return
        image = cv2.imread(self._suppression_input_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Reflection Suppression", "이미지를 읽을 수 없습니다.")
            return

        from calibration.windshield.reflection_suppression.config import DEFAULT_MAX_CORRECTION
        from ui.reflection_suppression_worker import ReflectionSuppressionWorker

        self.suppression_run_button.setEnabled(False)
        self.suppression_status_label.setText("Running reflection suppression...")
        worker = ReflectionSuppressionWorker(
            self._suppression_model_path, image, self._suppression_strength_value(), DEFAULT_MAX_CORRECTION,
        )
        thread = run_worker_in_thread(worker, self)
        worker.result_ready.connect(lambda result: self._on_suppression_finished(result, image))
        worker.error.connect(self._on_suppression_error)
        worker.progress.connect(self.suppression_status_label.setText)
        self._suppression_thread, self._suppression_worker = thread, worker
        thread.finished.connect(lambda: self.suppression_run_button.setEnabled(True))
        thread.start()

    def _on_suppression_error(self, message: str) -> None:
        self.suppression_status_label.setText(message)
        QMessageBox.critical(self, "Reflection Suppression", message)

    def _on_suppression_finished(self, result, original_image) -> None:
        self._suppression_result = result
        if not result.success:
            self.suppression_status_label.setText(result.error_message or "Suppression failed; original image returned.")
        elif result.skipped_due_to_low_reflection:
            self.suppression_status_label.setText(result.warning_message or "Suppression skipped (low reflection presence).")
        else:
            status = (
                f"Mean alpha {result.mean_alpha:.3f} (P95 {result.alpha_p95:.3f}, coverage "
                f"{result.alpha_coverage*100.0:.1f}%) · Mean correction {result.mean_correction:.4f} "
                f"· Strength {result.suppression_strength:.2f}"
            )
            if result.warning_message:
                status += f" · {result.warning_message}"
            self.suppression_status_label.setText(status)

        self._set_suppression_preview_image(self.suppression_original_image_label, original_image)
        if result.reflection_layer is not None:
            self._set_suppression_preview_image(self.suppression_reflection_image_label, result.reflection_layer)
        if result.alpha_map is not None:
            import numpy as np

            alpha_vis = np.clip(result.alpha_map * 255.0, 0, 255).astype(np.uint8)
            self._set_suppression_preview_image(self.suppression_alpha_image_label, alpha_vis, is_gray=True)
        if result.suppressed_image is not None:
            self._set_suppression_preview_image(self.suppression_output_image_label, result.suppressed_image)

        self._display_suppression_evaluation(result, original_image)

    def _display_suppression_evaluation(self, result, original_image) -> None:
        """STEP 6 evaluator를 suppression 전/후에 그대로 적용한다(사용자
        스펙 41번). Reference가 없으면 No-Reference Mode로 평가하고, 그
        결과를 절대 "실제 Reflection 측정값"처럼 표시하지 않는다(안정화
        라운드 항목 2) - Reference Mode에서만 Reflection Mean/P95/Coverage
        Reduction을, No-Reference Mode에서는 오직 Reflection Likelihood
        (heuristic, not ground truth)만 표시한다."""
        if not result.success or result.suppressed_image is None:
            for row in range(self.suppression_metrics_table.rowCount()):
                self.suppression_metrics_table.setItem(row, 0, QTableWidgetItem("N/A"))
                self.suppression_metrics_table.setItem(row, 1, QTableWidgetItem("N/A"))
            return

        import cv2

        from calibration.windshield.reflection.types import ReflectionEvaluationConfig
        from calibration.windshield.reflection_suppression.evaluation import evaluate_suppression

        reference_image = None
        if self._suppression_reference_path:
            reference_image = cv2.imread(self._suppression_reference_path, cv2.IMREAD_COLOR)
        cfg = ReflectionEvaluationConfig(mode="reference" if reference_image is not None else "no_reference")
        evaln = evaluate_suppression(original_image, result, reference_image=reference_image, config=cfg)

        is_reference = evaln.before.mode == "reference"
        first_row_label = "Reflection Mean" if is_reference else "Reflection Likelihood"
        self.suppression_metrics_table.setVerticalHeaderItem(0, QTableWidgetItem(first_row_label))

        def _pct(v):
            return f"{v*100.0:.1f}%" if v is not None else "N/A"

        if is_reference:
            before_vals = [evaln.before.reflection_mean, evaln.before.reflection_p95, evaln.before.reflection_coverage]
            after_vals = [evaln.after.reflection_mean, evaln.after.reflection_p95, evaln.after.reflection_coverage]
        else:
            before_vals = [evaln.reflection_likelihood_before, None, None]
            after_vals = [evaln.reflection_likelihood_after, None, None]

        for row, (b, a) in enumerate(zip(before_vals, after_vals)):
            self.suppression_metrics_table.setItem(row, 0, QTableWidgetItem(_pct(b)))
            self.suppression_metrics_table.setItem(row, 1, QTableWidgetItem(_pct(a)))

        self.suppression_metrics_table.setItem(3, 0, QTableWidgetItem("N/A"))
        self.suppression_metrics_table.setItem(
            3, 1, QTableWidgetItem(_pct(evaln.edge_retention_after) if evaln.edge_retention_after is not None else "N/A")
        )
        self.suppression_metrics_table.setItem(4, 0, QTableWidgetItem("N/A"))
        self.suppression_metrics_table.setItem(
            4, 1, QTableWidgetItem(_pct(evaln.contrast_retention_after) if evaln.contrast_retention_after is not None else "N/A")
        )
        self.suppression_metrics_table.setItem(5, 0, QTableWidgetItem("N/A"))
        self.suppression_metrics_table.setItem(
            5, 1, QTableWidgetItem(f"{evaln.over_suppression_score:.4f}" if evaln.over_suppression_score is not None else "N/A")
        )

        if not is_reference:
            note = " (no-reference heuristic, not ground truth)"
            if note not in self.suppression_status_label.text():
                self.suppression_status_label.setText(self.suppression_status_label.text() + note)

    def _set_suppression_preview_image(self, label: QLabel, image, is_gray: bool = False) -> None:
        import cv2
        from PySide6.QtGui import QImage, QPixmap

        if is_gray:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = rgb.copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            label.width() or 160, label.height() or 120, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

