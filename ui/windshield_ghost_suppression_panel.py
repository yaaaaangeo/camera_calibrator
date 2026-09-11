"""
camera_calibrator.ui.windshield_ghost_suppression_panel
==============================================================

Priority 7 안정화 - `ui/windshield_workspace.py`(God Object)에서 분리한
⑥ Ghost 탭의 Suppression sub-tab 전용 UI. Reflection Suppression 모델은
여기서 절대 재사용하지 않는다 - Ghost Suppression은 완전히 결정론적인
반복 재구성(`suppress_ghost`)이다(PyTorch 불필요).

`GhostSuppressionPanelMixin`은 `WindshieldWorkspace`에 다른 패널
mixin들과 함께 다중 상속되어 같은 `self`를 공유한다(ui/windshield_common.py
참고).

이 파일은 로직을 하나도 새로 추가/변경하지 않았다 - `ui/windshield_workspace.py`
에 있던 메서드를 그대로 옮겼을 뿐이다.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.windshield.ghost import GhostEvaluationConfig, fit_ghost_field_from_dataset, load_ghost_model, save_ghost_model
from calibration.windshield.ghost.config import DEFAULT_SPATIAL_COLS, DEFAULT_SPATIAL_ROWS
from ui.ghost_suppression_worker import GhostSuppressionWorker
from ui.windshield_common import ResponsiveRow, _ScrollTable, _fmt, configure_form_layout
from ui.worker import run_worker_in_thread


class GhostSuppressionPanelMixin:
    """⑥ Ghost 탭의 Suppression sub-tab 전용 UI."""

    def _build_ghost_suppression_subtab(self) -> QWidget:
        """STEP 8B - Ghost Suppression. Reflection Suppression 모델을 절대
        재사용하지 않는다(사용자 스펙 "Ghost 제거를 위해 Reflection
        Suppression 모델을 사용하지 않는다") - deterministic iterative
        reconstruction 결과만 불러와 inference/Before-After 평가를 한다."""
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("GHOST SUPPRESSION")
        form = QFormLayout(group)

        model_row = ResponsiveRow(breakpoint=900)
        self.ghost_suppression_model_path_label = QLabel("N/A")
        load_model_btn = QPushButton("Load Ghost Model...")
        load_model_btn.clicked.connect(self._on_load_ghost_suppression_model)
        # GhostField fit은 Point Source Evaluation 결과에서만 허용된다
        # (STEP 8 semantic/safety fix 3번) - Edge Target은 1D scalar
        # offset만 갖고 2D dx/dy field를 만들 수 없고, General Likelihood는
        # displacement 자체를 측정하지 않는 heuristic이다. 이미 저장된
        # GhostField를 "Load"해서 Suppression에 쓰는 것은 mode와 무관하게
        # 항상 허용한다(3-D번).
        self.ghost_fit_button = QPushButton("Fit From Last Evaluation")
        self.ghost_fit_button.clicked.connect(self._on_fit_ghost_model_from_evaluation)
        self.ghost_fit_button.setEnabled(False)
        self.ghost_fit_button.setToolTip(
            "GhostField fitting requires 2D main/ghost point displacement.\n"
            "Available only for Point Source mode."
        )
        save_btn = QPushButton("Save Model...")
        save_btn.clicked.connect(self._on_save_ghost_model)
        model_row.addWidget(load_model_btn)
        model_row.addWidget(self.ghost_fit_button)
        model_row.addWidget(save_btn)
        model_row.addWidget(self.ghost_suppression_model_path_label, stretch=1)
        form.addRow("Model:", model_row)

        input_row = ResponsiveRow(breakpoint=620)
        self.ghost_suppression_input_path_label = QLabel("N/A")
        load_input_btn = QPushButton("Load Image...")
        load_input_btn.clicked.connect(self._on_load_ghost_suppression_input_image)
        input_row.addWidget(load_input_btn)
        input_row.addWidget(self.ghost_suppression_input_path_label, stretch=1)
        form.addRow("Input:", input_row)
        self.ghost_suppression_model_path_label.setWordWrap(True)
        self.ghost_suppression_input_path_label.setWordWrap(True)
        configure_form_layout(form)
        layout.addWidget(group)

        action_row = ResponsiveRow(breakpoint=480)
        self.ghost_suppression_run_button = QPushButton("Run Suppression")
        self.ghost_suppression_run_button.clicked.connect(self._on_run_ghost_suppression)
        action_row.addWidget(self.ghost_suppression_run_button)
        action_row.addStretch(1)
        layout.addWidget(action_row)

        self.ghost_suppression_status_label = QLabel(
            "Deterministic iterative reconstruction: T_(k+1) = clip(I - alpha*W(T_k), 0, 1). "
            "Not a large CNN, not trained end-to-end."
        )
        self.ghost_suppression_status_label.setWordWrap(True)
        layout.addWidget(self.ghost_suppression_status_label)

        viz_group = QGroupBox("VISUALIZATION")
        viz_layout = QVBoxLayout(viz_group)
        viz_grid = ResponsiveRow(breakpoint=900)
        viz_layout.addWidget(viz_grid)
        self.ghost_suppression_original_image_label = QLabel("Original")
        self.ghost_suppression_predicted_image_label = QLabel("Predicted Ghost")
        self.ghost_suppression_correction_image_label = QLabel("Correction Map")
        self.ghost_suppression_output_image_label = QLabel("Suppressed")
        for lbl in (
            self.ghost_suppression_original_image_label,
            self.ghost_suppression_predicted_image_label,
            self.ghost_suppression_correction_image_label,
            self.ghost_suppression_output_image_label,
        ):
            lbl.setMinimumSize(160, 120)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid gray;")
            viz_grid.addWidget(lbl)
        layout.addWidget(viz_group)

        # STEP 8A evaluator를 그대로 재사용한 Before/After 비교(사용자 스펙
        # "Must reuse the exact same STEP 8A evaluator for Before/After
        # comparison").
        self.ghost_suppression_metrics_table = _ScrollTable(7, 2)
        self.ghost_suppression_metrics_table.setHorizontalHeaderLabels(["Before", "After"])
        self.ghost_suppression_metrics_table.setVerticalHeaderLabels([
            "Ghost Strength Mean", "Detection Count", "Mean Correction", "Max Correction",
            # Ghost만 지우는 것으로는 성공이 아니다(사용자 스펙 7-E번) -
            # Detail Retention / Over-Suppression을 항상 나란히 보여준다.
            "Edge Retention", "Clean Region Change", "Over-Suppression Score",
        ])
        layout.addWidget(self.ghost_suppression_metrics_table)
        layout.addStretch(1)
        return page

    def _on_load_ghost_suppression_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Ghost Model", "", "Ghost Model (*.yml *.yaml)")
        if not path:
            return
        try:
            field = load_ghost_model(path)
        except Exception as e:  # noqa: BLE001 - shown directly in the UI
            QMessageBox.critical(self, "Ghost Suppression", f"모델을 불러오지 못했습니다: {e}")
            return
        self._ghost_suppression_model_path = path
        self._ghost_suppression_field = field
        self.ghost_suppression_model_path_label.setText(path)
        # project save 시 이 모델도 함께 저장되도록 등록한다(STEP 8
        # stabilization 4번) - YAML 저장 여부와 무관하게 `.ccproj`에 남는다.
        self._ghost_models[Path(path).stem] = field

    def _on_fit_ghost_model_from_evaluation(self) -> None:
        """Dataset 전체(모든 frame의 detection)에서 `GhostField`를 fit한다
        (STEP 8 stabilization 3-G번, "반드시 per_frame[0]를 사용하지
        않는다") - 단일 이미지만 평가했다면 dataset은 frame 1개짜리로
        취급되어 결과가 기존 constant fit과 동일하다."""
        if self._ghost_result is None or not self._ghost_result.per_frame:
            QMessageBox.warning(self, "Ghost Suppression", "먼저 Evaluation 탭에서 Ghost Evaluation을 실행하세요.")
            return
        if self._ghost_result.mode != "point_source":
            # 버튼이 비활성화되어 있어야 정상이지만(3-C번), 방어적으로 한 번
            # 더 확인한다 - backend guard(fit_ghost_field_from_dataset의
            # ValueError)에만 기대지 않는다.
            QMessageBox.warning(
                self, "Ghost Suppression",
                "GhostField는 Point Source Evaluation 결과에서만 fit할 수 있습니다.",
            )
            return
        source_path = self._ghost_image_path or (
            self._ghost_dataset_image_paths(self._ghost_dataset_dir)[0] if self._ghost_dataset_dir else ""
        )
        if not source_path:
            QMessageBox.warning(self, "Ghost Suppression", "Evaluation에 쓰인 이미지 크기를 알 수 없습니다.")
            return
        import cv2

        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Ghost Suppression", "이미지를 읽을 수 없습니다.")
            return
        h, w = image.shape[:2]
        field = fit_ghost_field_from_dataset(
            self._ghost_result, image_width=w, image_height=h, rows=DEFAULT_SPATIAL_ROWS, cols=DEFAULT_SPATIAL_COLS,
        )
        self._ghost_suppression_field = field
        self._ghost_suppression_model_path = ""
        self.ghost_suppression_model_path_label.setText("(fitted from dataset, not saved to YAML)")
        model_key = f"{self._ghost_result.mode}_fitted"
        self._ghost_models[model_key] = field

    def _on_save_ghost_model(self) -> None:
        field = getattr(self, "_ghost_suppression_field", None)
        if field is None:
            QMessageBox.warning(self, "Ghost Suppression", "먼저 'Fit From Last Evaluation'으로 model을 만드세요.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Ghost Model", "ghost_model.yml", "Ghost Model (*.yml *.yaml)")
        if not path:
            return
        save_ghost_model(field, path)
        self._ghost_suppression_model_path = path
        self.ghost_suppression_model_path_label.setText(path)

    def _on_load_ghost_suppression_input_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Ghost Suppression Input Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._ghost_suppression_input_path = path
        self.ghost_suppression_input_path_label.setText(path)

    def _on_run_ghost_suppression(self) -> None:
        import cv2

        ghost_field = getattr(self, "_ghost_suppression_field", None)
        if ghost_field is None and not self._ghost_suppression_model_path:
            QMessageBox.warning(self, "Ghost Suppression", "Model을 불러오거나 마지막 Evaluation에서 fit하세요.")
            return
        if not self._ghost_suppression_input_path:
            QMessageBox.warning(self, "Ghost Suppression", "Input image가 필요합니다.")
            return
        image = cv2.imread(self._ghost_suppression_input_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Ghost Suppression", "이미지를 읽을 수 없습니다.")
            return

        # Before/After 평가는 마지막 Evaluation에 쓰인 mode를 그대로
        # 따른다(사용자 스펙 8번, mode-aware dispatcher) - 하드코딩된
        # point_source로 고정하지 않는다.
        eval_mode = self._ghost_result.mode if self._ghost_result is not None else "point_source"
        eval_cfg = GhostEvaluationConfig(mode=eval_mode, edge_axis=self._ghost_edge_axis())
        worker = GhostSuppressionWorker(
            self._ghost_suppression_model_path or None,
            image,
            eval_cfg,
            ghost_field=ghost_field,
        )
        thread = run_worker_in_thread(worker, self)
        self.ghost_suppression_run_button.setEnabled(False)
        self.ghost_suppression_status_label.setText("Running ghost suppression...")
        worker.result_ready.connect(lambda result: self._on_ghost_suppression_finished(result, image))
        worker.error.connect(self._on_ghost_suppression_error)
        worker.progress.connect(self.ghost_suppression_status_label.setText)
        self._ghost_suppression_thread, self._ghost_suppression_worker = thread, worker
        thread.finished.connect(lambda: self.ghost_suppression_run_button.setEnabled(True))
        thread.start()

    def _on_ghost_suppression_error(self, message: str) -> None:
        self.ghost_suppression_status_label.setText(message)
        QMessageBox.critical(self, "Ghost Suppression", message)

    def _on_ghost_suppression_finished(self, result_tuple, original_image) -> None:
        supp, evaln = result_tuple
        self._ghost_suppression_result = supp
        if not supp.success:
            self.ghost_suppression_status_label.setText(supp.warning_message or supp.error_message or "Suppression failed; original image returned.")
        else:
            self.ghost_suppression_status_label.setText(
                f"Iterations {supp.iterations} | Mean correction {supp.mean_correction:.4f} | Max correction {supp.max_correction:.4f}"
            )

        self._set_suppression_preview_image(self.ghost_suppression_original_image_label, original_image)
        if supp.predicted_ghost_image is not None:
            import numpy as np
            pred_u8 = np.clip(supp.predicted_ghost_image, 0, 255).astype(np.uint8)
            self._set_suppression_preview_image(self.ghost_suppression_predicted_image_label, pred_u8)
        if supp.correction_map is not None:
            import numpy as np
            corr_gray = np.clip(np.mean(supp.correction_map, axis=-1) if supp.correction_map.ndim == 3 else supp.correction_map, 0, 255).astype(np.uint8)
            self._set_suppression_preview_image(self.ghost_suppression_correction_image_label, corr_gray, is_gray=True)
        if supp.suppressed_image is not None:
            self._set_suppression_preview_image(self.ghost_suppression_output_image_label, supp.suppressed_image)

        before, after = evaln.before, evaln.after
        # Mode별 primary metric을 절대 섞지 않는다(STEP 8 semantic fix
        # 2번) - General(No-Reference) Likelihood는 Strength Reduction/
        # Detection Reduction이 아니라 Likelihood Before/After/Reduction을
        # primary로 보여준다. General mode의 detection_count는 heuristic
        # profile 개수일 뿐 실제 ghost object 개수가 아니므로(사용자 스펙
        # 2-F번) 그 행 대신 Likelihood Reduction을 보여준다.
        is_general = before.mode == "general_likelihood"
        if is_general:
            self.ghost_suppression_metrics_table.setVerticalHeaderItem(0, QTableWidgetItem("Ghost Likelihood"))
            self.ghost_suppression_metrics_table.setVerticalHeaderItem(1, QTableWidgetItem("Likelihood Reduction"))
            self.ghost_suppression_metrics_table.setItem(0, 0, QTableWidgetItem(_fmt(before.ghost_likelihood)))
            self.ghost_suppression_metrics_table.setItem(0, 1, QTableWidgetItem(_fmt(after.ghost_likelihood)))
            self.ghost_suppression_metrics_table.setItem(1, 0, QTableWidgetItem("N/A"))
            self.ghost_suppression_metrics_table.setItem(1, 1, QTableWidgetItem(_fmt(evaln.likelihood_reduction)))
        else:
            self.ghost_suppression_metrics_table.setVerticalHeaderItem(0, QTableWidgetItem("Ghost Strength Mean"))
            self.ghost_suppression_metrics_table.setVerticalHeaderItem(1, QTableWidgetItem("Detection Count"))
            self.ghost_suppression_metrics_table.setItem(0, 0, QTableWidgetItem(_fmt(before.mean_strength_ratio)))
            self.ghost_suppression_metrics_table.setItem(0, 1, QTableWidgetItem(_fmt(after.mean_strength_ratio)))
            self.ghost_suppression_metrics_table.setItem(1, 0, QTableWidgetItem(str(before.detection_count)))
            self.ghost_suppression_metrics_table.setItem(1, 1, QTableWidgetItem(str(after.detection_count)))
        self.ghost_suppression_metrics_table.setItem(2, 0, QTableWidgetItem("N/A"))
        self.ghost_suppression_metrics_table.setItem(2, 1, QTableWidgetItem(f"{supp.mean_correction:.4f}"))
        self.ghost_suppression_metrics_table.setItem(3, 0, QTableWidgetItem("N/A"))
        self.ghost_suppression_metrics_table.setItem(3, 1, QTableWidgetItem(f"{supp.max_correction:.4f}"))
        self.ghost_suppression_metrics_table.setItem(4, 0, QTableWidgetItem("N/A"))
        self.ghost_suppression_metrics_table.setItem(4, 1, QTableWidgetItem(_fmt(supp.edge_retention)))
        self.ghost_suppression_metrics_table.setItem(5, 0, QTableWidgetItem("N/A"))
        self.ghost_suppression_metrics_table.setItem(5, 1, QTableWidgetItem(_fmt(supp.clean_region_change)))
        self.ghost_suppression_metrics_table.setItem(6, 0, QTableWidgetItem("N/A"))
        self.ghost_suppression_metrics_table.setItem(6, 1, QTableWidgetItem(_fmt(supp.over_suppression_score)))
