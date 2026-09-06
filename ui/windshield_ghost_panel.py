"""
camera_calibrator.ui.windshield_ghost_panel
==============================================================

Priority 7 안정화 - `ui/windshield_workspace.py`(God Object)에서 분리한
⑥ Ghost 탭의 Evaluation sub-tab 전용 UI. Ghost Suppression은 별도 파일
(ui/windshield_ghost_suppression_panel.py)이 담당한다 - Reflection과
완전히 독립된 상태/worker/결과 타입을 쓴다(사용자 스펙 "Ghost는 기존
Reflection 기능 안에 넣지 않는다").

`GhostPanelMixin`은 `WindshieldWorkspace`에 다른 패널 mixin들과 함께
다중 상속되어 같은 `self`를 공유한다(ui/windshield_common.py 참고) -
`_build_ghost_tab()`이 여기서 `self._build_ghost_suppression_subtab()`
(다른 mixin에 정의됨)을 호출하지만 Python MRO가 찾아준다.

이 파일은 로직을 하나도 새로 추가/변경하지 않았다 - `ui/windshield_workspace.py`
에 있던 메서드를 그대로 옮겼을 뿐이다. Displacement(vector field)와
Strength(heatmap)는 절대 하나의 표/이미지로 합치지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
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

from calibration.windshield.ghost import GhostDatasetResult, GhostEvaluationConfig
from calibration.windshield.ghost.config import DEFAULT_SPATIAL_COLS, DEFAULT_SPATIAL_ROWS, GHOST_DATASET_IMAGE_EXTENSIONS
from ui.ghost_evaluation_worker import GhostEvaluationWorker
from ui.windshield_common import _ScrollTable, _fit_table_to_rows, _fmt
from ui.worker import run_worker_in_thread


class GhostPanelMixin:
    """⑥ Ghost 탭 컨테이너 + Evaluation sub-tab 전용 UI."""

    def _build_ghost_tab(self) -> QWidget:
        outer = QTabWidget()
        outer.addTab(self._build_ghost_evaluation_subtab(), "Evaluation")
        outer.addTab(self._build_ghost_suppression_subtab(), "Suppression")
        return outer

    def _build_ghost_evaluation_subtab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("GHOST / DOUBLE IMAGE EVALUATION")
        group_layout = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        self._ghost_mode_button_group = QButtonGroup(self)
        self.ghost_point_source_radio = QRadioButton("Point Source")
        self.ghost_point_source_radio.setChecked(True)
        self.ghost_edge_target_radio = QRadioButton("Edge Target")
        self.ghost_general_likelihood_radio = QRadioButton("General Image (Likelihood)")
        for radio in (
            self.ghost_point_source_radio,
            self.ghost_edge_target_radio,
            self.ghost_general_likelihood_radio,
        ):
            self._ghost_mode_button_group.addButton(radio)
            mode_row.addWidget(radio)
        mode_row.addStretch(1)
        group_layout.addLayout(mode_row)

        edge_axis_row = QHBoxLayout()
        edge_axis_row.addWidget(QLabel("Edge Axis (Edge Target only):"))
        self.ghost_edge_axis_combo = QComboBox()
        self.ghost_edge_axis_combo.addItems(["Auto", "Vertical", "Horizontal"])
        self.ghost_edge_axis_combo.setCurrentText("Vertical")
        edge_axis_row.addWidget(self.ghost_edge_axis_combo)
        edge_axis_row.addStretch(1)
        group_layout.addLayout(edge_axis_row)

        image_row = QHBoxLayout()
        self.ghost_image_path_label = QLabel("N/A")
        load_image_btn = QPushButton("Load Image...")
        load_image_btn.clicked.connect(self._on_load_ghost_image)
        load_dataset_btn = QPushButton("Load Dataset Directory...")
        load_dataset_btn.clicked.connect(self._on_load_ghost_dataset_directory)
        image_row.addWidget(load_image_btn)
        image_row.addWidget(load_dataset_btn)
        image_row.addWidget(self.ghost_image_path_label, stretch=1)
        group_layout.addLayout(image_row)

        action_row = QHBoxLayout()
        self.ghost_run_button = QPushButton("Run Evaluation")
        self.ghost_run_button.clicked.connect(self._on_run_ghost_evaluation)
        action_row.addWidget(self.ghost_run_button)
        action_row.addStretch(1)
        group_layout.addLayout(action_row)

        self.ghost_status_label = QLabel(
            "Ghost = same exterior scene shifted/warped by internal windshield multi-reflection. "
            "Not related to Reflection(interior scene overlay)."
        )
        self.ghost_status_label.setWordWrap(True)
        group_layout.addWidget(self.ghost_status_label)
        layout.addWidget(group)

        # General(No-Reference) Likelihood 모드는 Point Source/Edge Target과
        # 의미가 다르므로(사용자 스펙 6-A번) 전용 label로 완전히 분리해
        # 보여준다 - Ground Truth라는 단어를 절대 쓰지 않는다.
        self.ghost_likelihood_label = QLabel("")
        self.ghost_likelihood_label.setWordWrap(True)
        self.ghost_likelihood_label.setStyleSheet("font-weight: 600;")
        self.ghost_likelihood_label.setVisible(False)
        layout.addWidget(self.ghost_likelihood_label)

        self.ghost_metrics_table = _ScrollTable(9, 1)
        self.ghost_metrics_table.setHorizontalHeaderLabels(["Value"])
        self.ghost_metrics_table.setVerticalHeaderLabels([
            "Mode", "Detection Count", "Detection Rate",
            "Ghost Offset X [px]", "Ghost Offset Y [px]",
            "Ghost Distance (Median) [px]", "Ghost Distance (P95) [px]",
            "Ghost Strength Mean", "Ghost Strength P95",
        ])
        layout.addWidget(self.ghost_metrics_table)

        # Point Source(2D dx/dy)와 Edge Target(1D scalar offset)을 같은
        # row에 억지로 겹쳐 보여주지 않는다(사용자 스펙 2-F번) - Edge 전용
        # 행을 별도 table로 분리한다. General Likelihood 모드에서는 두
        # table 모두 의미가 없으므로 숨긴다(사용자 스펙 6-A번).
        self.ghost_edge_metrics_table = _ScrollTable(2, 1)
        self.ghost_edge_metrics_table.setHorizontalHeaderLabels(["Value"])
        self.ghost_edge_metrics_table.setVerticalHeaderLabels(["Edge Ghost Offset Median [px]", "Edge Ghost Offset P95 [px]"])
        layout.addWidget(self.ghost_edge_metrics_table)

        # Overlay(Main/Ghost point + arrow) - 사용자 스펙 6-C번.
        overlay_group = QGroupBox("MAIN / GHOST OVERLAY")
        overlay_layout = QVBoxLayout(overlay_group)
        self.ghost_overlay_image_label = QLabel("N/A")
        self.ghost_overlay_image_label.setMinimumSize(320, 200)
        self.ghost_overlay_image_label.setAlignment(Qt.AlignCenter)
        self.ghost_overlay_image_label.setStyleSheet("border: 1px solid gray;")
        overlay_layout.addWidget(self.ghost_overlay_image_label)
        layout.addWidget(overlay_group)

        # Displacement(vector field)와 Strength(heatmap)를 절대 하나의
        # 그림/표로 합치지 않는다(사용자 스펙 "Displacement와 Strength를
        # 한 map에 억지로 합치지 않는다") - 완전히 분리된 두 이미지 +
        # 두 표로 보여준다.
        viz_group = QGroupBox("SPATIAL GHOST MAP (never merged: displacement vector field vs strength heatmap)")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.addWidget(QLabel("Displacement Vector Field (arrow = direction/magnitude, mean dx/dy per cell):"))
        self.ghost_vector_field_image_label = QLabel("N/A")
        self.ghost_vector_field_image_label.setMinimumSize(240, 160)
        self.ghost_vector_field_image_label.setAlignment(Qt.AlignCenter)
        self.ghost_vector_field_image_label.setStyleSheet("border: 1px solid gray;")
        viz_layout.addWidget(self.ghost_vector_field_image_label)
        self.ghost_vector_field_table = _ScrollTable(DEFAULT_SPATIAL_ROWS, DEFAULT_SPATIAL_COLS)
        self.ghost_vector_field_table.setHorizontalHeaderLabels([f"C{c+1}" for c in range(DEFAULT_SPATIAL_COLS)])
        self.ghost_vector_field_table.setVerticalHeaderLabels([f"R{r+1}" for r in range(DEFAULT_SPATIAL_ROWS)])
        viz_layout.addWidget(self.ghost_vector_field_table)

        viz_layout.addWidget(QLabel("Strength Heatmap (color = mean strength ratio per cell):"))
        self.ghost_strength_heatmap_image_label = QLabel("N/A")
        self.ghost_strength_heatmap_image_label.setMinimumSize(240, 160)
        self.ghost_strength_heatmap_image_label.setAlignment(Qt.AlignCenter)
        self.ghost_strength_heatmap_image_label.setStyleSheet("border: 1px solid gray;")
        viz_layout.addWidget(self.ghost_strength_heatmap_image_label)
        self.ghost_strength_heatmap_table = _ScrollTable(DEFAULT_SPATIAL_ROWS, DEFAULT_SPATIAL_COLS)
        self.ghost_strength_heatmap_table.setHorizontalHeaderLabels([f"C{c+1}" for c in range(DEFAULT_SPATIAL_COLS)])
        self.ghost_strength_heatmap_table.setVerticalHeaderLabels([f"R{r+1}" for r in range(DEFAULT_SPATIAL_ROWS)])
        viz_layout.addWidget(self.ghost_strength_heatmap_table)
        layout.addWidget(viz_group)
        layout.addStretch(1)
        return page

    def _on_load_ghost_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Ghost Evaluation Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._ghost_image_path = path
        self._ghost_dataset_dir = ""
        self.ghost_image_path_label.setText(path)

    def _on_load_ghost_dataset_directory(self) -> None:
        """STEP 8 stabilization 3-A번 - 단일 이미지 대신 dataset 디렉토리
        전체를 Ghost Evaluation에 쓸 수 있게 한다."""
        directory = QFileDialog.getExistingDirectory(self, "Load Ghost Dataset Directory")
        if not directory:
            return
        self._ghost_dataset_dir = directory
        self._ghost_image_path = ""
        self.ghost_image_path_label.setText(f"[dataset] {directory}")

    def _ghost_dataset_image_paths(self, directory: str) -> list[str]:
        d = Path(directory)
        paths = sorted(
            str(p) for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in GHOST_DATASET_IMAGE_EXTENSIONS
        )
        return paths

    def _ghost_mode(self) -> str:
        if self.ghost_edge_target_radio.isChecked():
            return "edge_target"
        if self.ghost_general_likelihood_radio.isChecked():
            return "general_likelihood"
        return "point_source"

    def _ghost_edge_axis(self) -> str:
        return self.ghost_edge_axis_combo.currentText().lower()

    def _on_run_ghost_evaluation(self) -> None:
        if self._ghost_dataset_dir:
            image_paths = self._ghost_dataset_image_paths(self._ghost_dataset_dir)
            frame_ids = [Path(p).stem for p in image_paths]
            if not image_paths:
                QMessageBox.warning(self, "Ghost Evaluation", "디렉토리에서 이미지 파일을 찾지 못했습니다.")
                return
        elif self._ghost_image_path:
            image_paths = [self._ghost_image_path]
            frame_ids = [Path(self._ghost_image_path).stem]
        else:
            QMessageBox.warning(self, "Ghost Evaluation", "Image 또는 Dataset Directory가 필요합니다.")
            return

        mode = self._ghost_mode()
        cfg = GhostEvaluationConfig(mode=mode, edge_axis=self._ghost_edge_axis())
        worker = GhostEvaluationWorker(image_paths, cfg, frame_ids=frame_ids)
        thread = run_worker_in_thread(worker, self)
        self.ghost_run_button.setEnabled(False)
        self.ghost_status_label.setText("Ghost evaluation running...")
        worker.result_ready.connect(self._on_ghost_evaluation_finished)
        worker.error.connect(self._on_ghost_evaluation_error)
        worker.progress.connect(self.ghost_status_label.setText)
        self._ghost_thread, self._ghost_worker = thread, worker
        thread.finished.connect(lambda: self.ghost_run_button.setEnabled(True))
        thread.start()

    def _on_ghost_evaluation_error(self, message: str) -> None:
        self.ghost_status_label.setText(message)
        QMessageBox.critical(self, "Ghost Evaluation", message)

    def _on_ghost_evaluation_finished(self, result: GhostDatasetResult) -> None:
        self._ghost_result = result
        self._ghost_results["latest"] = result
        self._display_ghost_result(result)

    def _display_ghost_result(self, dataset_result: GhostDatasetResult) -> None:
        frame = dataset_result.per_frame[0] if dataset_result.per_frame else None
        if frame is None:
            self.ghost_status_label.setText(dataset_result.error_message or dataset_result.warning_message or "No ghost result.")
            return

        status = f"Metric v{frame.metric_version} | {len(dataset_result.per_frame)} frame(s)"
        if frame.warning_message:
            status += f" | {frame.warning_message}"
        self.ghost_status_label.setText(status)

        # General Likelihood는 Point Source/Edge Target과 의미가 완전히
        # 다르다(사용자 스펙 6-A번) - 전용 label만 보여주고, 의미 없는
        # dx/dy/거리/edge 행은 아예 숨긴다.
        is_general = frame.is_likelihood
        is_edge = frame.mode == "edge_target"
        self.ghost_likelihood_label.setVisible(is_general)
        self.ghost_metrics_table.setVisible(not is_general)
        self.ghost_edge_metrics_table.setVisible(is_edge and not is_general)

        # GhostField fit은 Point Source 결과에서만 허용한다(STEP 8
        # semantic/safety fix 3-C번) - 버튼 자체를 비활성화해 실수로라도
        # Edge/General 결과에서 fit을 시도하지 못하게 한다.
        is_point_source = frame.mode == "point_source"
        self.ghost_fit_button.setEnabled(is_point_source)
        self.ghost_fit_button.setToolTip(
            "Fit a GhostField from the current dataset evaluation."
            if is_point_source else
            "GhostField fitting requires 2D main/ghost point displacement.\n"
            "Available only for Point Source mode."
        )
        if is_general:
            # Dataset aggregate는 반드시 mean_ghost_likelihood/median_ghost_likelihood/
            # p95_ghost_likelihood를 쓴다 - mean_strength(2차 edge의 상대
            # 강도)는 Likelihood(double-edge 패턴이 나타난 비율)와 다른
            # 값이므로 여기 대신 쓰지 않는다(STEP 8 semantic fix 1번).
            self.ghost_likelihood_label.setText(
                "Ghost Likelihood\n"
                f"Current Frame   {_fmt(frame.ghost_likelihood)}\n"
                f"Dataset Mean    {_fmt(dataset_result.mean_ghost_likelihood)}\n"
                f"Dataset Median  {_fmt(dataset_result.median_ghost_likelihood)}\n"
                f"Dataset P95     {_fmt(dataset_result.p95_ghost_likelihood)}\n\n"
                "No-reference heuristic - NOT Ground Truth."
            )

        values = [
            frame.mode,
            str(frame.detection_count),
            f"{frame.detection_rate * 100.0:.1f}%" if frame.detection_rate is not None else "N/A",
            _fmt(frame.mean_offset_x_px),
            _fmt(frame.mean_offset_y_px),
            _fmt(frame.median_distance_px),
            _fmt(frame.p95_distance_px),
            _fmt(frame.mean_strength_ratio),
            _fmt(frame.p95_strength_ratio),
        ]
        for row, v in enumerate(values):
            self.ghost_metrics_table.setItem(row, 0, QTableWidgetItem(str(v)))
        _fit_table_to_rows(self.ghost_metrics_table)

        self.ghost_edge_metrics_table.setItem(0, 0, QTableWidgetItem(_fmt(frame.edge_offset_median_px)))
        self.ghost_edge_metrics_table.setItem(1, 0, QTableWidgetItem(_fmt(frame.edge_offset_p95_px)))

        rows = self.ghost_vector_field_table.rowCount()
        cols = self.ghost_vector_field_table.columnCount()
        for r in range(rows):
            for c in range(cols):
                self.ghost_vector_field_table.setItem(r, c, QTableWidgetItem("N/A"))
                self.ghost_strength_heatmap_table.setItem(r, c, QTableWidgetItem("N/A"))
        for cell in frame.spatial_map:
            if cell.sample_count <= 0 or cell.row >= rows or cell.col >= cols:
                continue
            dx = f"{cell.mean_offset_x_px:.1f}" if cell.mean_offset_x_px is not None else "?"
            dy = f"{cell.mean_offset_y_px:.1f}" if cell.mean_offset_y_px is not None else "?"
            self.ghost_vector_field_table.setItem(cell.row, cell.col, QTableWidgetItem(f"({dx},{dy})"))
            strength_text = f"{cell.mean_strength_ratio:.3f}" if cell.mean_strength_ratio is not None else "N/A"
            self.ghost_strength_heatmap_table.setItem(cell.row, cell.col, QTableWidgetItem(strength_text))

        # 실제 시각화(사용자 스펙 6-C/6-D/6-E번) - 전부 이미 계산된
        # detections/spatial_map을 그리기만 한다(재분석 없음).
        self._update_ghost_visualizations(frame)

    def _update_ghost_visualizations(self, frame) -> None:
        from calibration.windshield.ghost.visualization import (
            render_ghost_point_overlay,
            render_strength_heatmap_image,
            render_vector_field_image,
        )

        if frame.mode == "point_source" and frame.detections:
            import cv2

            source_path = self._ghost_image_path or (
                self._ghost_dataset_image_paths(self._ghost_dataset_dir)[0] if self._ghost_dataset_dir else ""
            )
            base_image = cv2.imread(source_path, cv2.IMREAD_COLOR) if source_path else None
            if base_image is not None:
                overlay = render_ghost_point_overlay(base_image, frame.detections)
                self._set_suppression_preview_image(self.ghost_overlay_image_label, overlay)

        if frame.spatial_map:
            rows = self.ghost_vector_field_table.rowCount()
            cols = self.ghost_vector_field_table.columnCount()
            vf_img = render_vector_field_image(frame.spatial_map, rows=rows, cols=cols)
            self._set_suppression_preview_image(self.ghost_vector_field_image_label, vf_img)
            heat_img = render_strength_heatmap_image(frame.spatial_map, rows=rows, cols=cols)
            self._set_suppression_preview_image(self.ghost_strength_heatmap_image_label, heat_img)

    # ------------------------------------------------------------------
    # STEP 8B - Ghost Suppression handlers
    # ------------------------------------------------------------------

